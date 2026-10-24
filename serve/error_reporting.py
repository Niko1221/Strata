"""Error reporting for the server process. Opt-in: inert without a DSN.

Only exceptions that reach this module are reported. Most handlers keep their exception to
themselves, so what is reported is the propagating set, not everything that failed. The engine
itself reports nothing; its failures live in stderr and the log. `sentry-sdk` is not a runtime
requirement: with a DSN set and the package absent, one line is printed and the server runs as
it did before.
"""

from __future__ import annotations

import os
import sys

_sdk = None
active = False


def dsn(cfg=None) -> str:
    report = (cfg or {}).get("error_reporting") or {}
    return (os.environ.get("STRATA_GLITCHTIP_DSN") or report.get("dsn") or "").strip()


def init(cfg=None) -> bool:
    """Start reporting when the config or the environment names a DSN; otherwise do nothing."""
    global _sdk, active
    got = dsn(cfg)
    if not got:
        return False
    try:
        import sentry_sdk
    except ImportError:
        sys.stderr.write("strata: error reporting is configured but sentry-sdk is not installed; nothing is reported\n")
        return False
    report = (cfg or {}).get("error_reporting") or {}
    sentry_sdk.init(
        dsn=got,
        environment=os.environ.get("STRATA_GLITCHTIP_ENV") or report.get("environment") or "default",
        release=os.environ.get("STRATA_RELEASE") or report.get("release") or None,
        send_default_pii=False,          # request bodies carry user text; the report is about the failure
        traces_sample_rate=0.0,          # an error tracker, not a sampling pipeline
        shutdown_timeout=3,              # a flush must not hang a shutdown
    )
    _sdk, active = sentry_sdk, True
    return True


def capture(exc) -> bool:
    """Report one exception a handler would otherwise keep to itself. A client hanging up is not a failure."""
    if not active or _sdk is None or exc is None or isinstance(exc, ConnectionError):
        return False
    try:
        _sdk.capture_exception(exc)
        return True
    except Exception:
        return False
