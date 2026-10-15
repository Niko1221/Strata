"""Optional Hermes plugin, installed separately; never patches Hermes core."""
import json
import threading

from .broker import AdmissionError, EngineClient, SupervisorBroker


def register(ctx):
    from agent.tool_execution_context import current_tool_execution_context
    from agent.secret_scope import get_secret
    from hermes_cli.authorized_tool_execution import hold_foreground_execution
    from tools.interrupt import is_interrupted

    brokers, lock = {}, threading.Lock()
    secret_name = ctx.get_config("token_env", "STRATA_RESOURCE_LEASE_TOKEN")
    ctx.register_private_env_keys([secret_name])

    def enabled():
        return ctx.get_config("enabled", False) is True

    def broker(scope):
        home = scope.get("profile_home")
        if not home:
            raise AdmissionError("Resource handoff requires a profile-scoped supervisor")
        url = ctx.get_config("base_url", "http://127.0.0.1:8080")
        profiles = ctx.get_config("profiles", {})
        defaults = ctx.get_config("default_profiles", [])
        if ctx.get_config("token_env", "STRATA_RESOURCE_LEASE_TOKEN") != secret_name:
            raise AdmissionError("Reload the resource-feedback plugin after changing token_env")
        key = (home, url, json.dumps(profiles, sort_keys=True), tuple(defaults), secret_name)
        with lock:
            if key not in brokers:
                def client_factory():
                    return EngineClient(url, get_secret(secret_name))
                brokers[key] = SupervisorBroker(client_factory, profiles, default_profiles=defaults,
                                               wait_seconds=ctx.get_config("wait_seconds", 180))
            return brokers[key]

    def execution(*, tool_name, args, next_call, **context):
        if not enabled() or tool_name != "terminal":
            return next_call()
        scope = dict(current_tool_execution_context())
        command = args.get("command")
        if not isinstance(command, str):
            raise AdmissionError("Terminal command is missing")
        owner = broker(scope)
        if owner.match(command) is None:
            return next_call()
        def protected_call():
            with hold_foreground_execution():
                return next_call()
        return owner.run(scope, command, protected_call,
            background=args.get("background", False), env_type=context.get("env_type", ""),
            cancelled=is_interrupted)

    def plan(args, **kwargs):
        try:
            if not enabled():
                raise AdmissionError("Resource feedback is disabled")
            scope = dict(current_tool_execution_context())
            result = broker(scope).plan(scope, args.get("profiles"))
            return json.dumps({"success": True, **result})
        except (AdmissionError, ValueError) as exc:
            return json.dumps({"success": False, "error": str(exc)})

    ctx.register_middleware("authorized_tool_execution", execution)
    ctx.register_tool(name="resource_plan", toolset="resource_feedback", check_fn=enabled,
        description="Select configured resource profiles for this supervisor's local tool work.",
        schema={"name": "resource_plan", "description":
            "Supervisor only: permit operator-configured resource classes before delegating local builds or GPU tests. "
            "Workers report unmet needs to you. This does not approve shell commands or allocate memory. "
            "An empty list revokes future grants. Tool work remains sequential within the resource broker.",
            "parameters": {"type": "object", "properties": {"profiles": {"type": "array", "items": {"type": "string"}}},
                           "required": ["profiles"], "additionalProperties": False}}, handler=plan)
