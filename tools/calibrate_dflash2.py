"""Fresh-request profile search for setup's opt-in DFlash2 calibration.

Prompt size changes allocations and expert residency. Compare whole profiles,
then confirm the winner against the baseline with independent engine starts.
"""
from __future__ import annotations

import math
import statistics
import threading
import time

PREFILLS = (256, 512, 1024)
FLOORS = (0.5, 0.7, 0.85)
MIN_FREE_MIB = 1024                 # startup headroom for lazy prefill/verification allocations
LONG_TOKENS = (2048, 4096)


def measure_profiles(args, settings, ids_list, make_long, start_engine, *, say=print,
                     progress=None, cancel_event=None, baseline_args=None, calibration=None):
    if calibration is None:
        import calibrate as calibration
    C = calibration

    report = {"objective": "balanced short/long normalized fresh-request time",
              "profiles": {}, "failed": {}, "confirmation": {}, "selected": "baseline"}
    tuned = C.apply(args, settings)
    context = int(C.arg_value(tuned, "--max-context") or 8192)
    budget = context - C.MAX_NEW - 256  # doc_prompt's chat header and tail also need room
    if budget < 512 or "--layer-split" in tuned:
        report["skipped"] = "needs one GPU and room for a fresh long prompt"
        return {}, report
    lengths = sorted({min(n, budget) for n in LONG_TOKENS})
    counter = 0
    default = None
    profiles = {}

    def checked():
        C._check_cancel(cancel_event)

    def failure(name, exc):
        why = str(exc).strip().splitlines()[0][:300] if str(exc).strip() else type(exc).__name__
        report["failed"][name] = why
        say(f"    DFlash2 profile {name} dropped: {why}")

    def start_safe(profile_args, name):
        """Never lower a requested reserve or increase an explicit expert cap.
        Auto caches get a larger reserve; explicit caches can only shrink.
        Unknown memory figures cannot qualify a new profile as safe.
        """
        current = list(profile_args)
        for attempt in range(3):
            checked()
            eng = start_engine(current)
            try:
                info = dict(eng.info or {})
                free = int(info["vram_free_mib"])
                if free >= MIN_FREE_MIB:
                    return eng, current, info
                missing = MIN_FREE_MIB - free + 256
                cap = C.arg_value(current, "--expert-cache")
                if cap is not None and cap != "auto":
                    slots, mib = int(info["expert_slots"]), float(info["expert_cache_mib"])
                    if slots < 1 or mib <= 0:
                        raise ValueError("cannot lower the expert cache from these memory figures")
                    count = slots - math.ceil(missing * slots / mib)
                    if count < 1:
                        raise ValueError("not enough VRAM for a safe expert-cache profile")
                    current = C.with_arg(current, "--expert-cache", str(min(int(cap), count)))
                else:
                    reserve = int(C.arg_value(current, "--vram-reserve-mib") or 600)
                    current = C.with_arg(current, "--vram-reserve-mib", str(reserve + missing))
            except BaseException:
                C.close(eng)
                raise
            C.close(eng)
        raise ValueError("startup headroom remains below the calibration target")

    def sample(eng, tune, full=False):
        nonlocal counter
        cases = [("short", ids) for ids in (ids_list if full else ids_list[:1])]
        for n in (lengths if full else lengths[:1]):
            counter += 1
            ids = make_long(counter, n)
            if len(ids) + C.MAX_NEW > context:
                raise ValueError("calibration prompt exceeds the configured context")
            cases.append(("long", ids))
        readings = []
        for kind, ids in cases:
            checked()
            started = time.monotonic()
            sampling = {"temperature": 0, "seed": 1234, "strata_checkpoint": False,
                        "strata_tune": tune}
            n = sum(t is not None for t in eng.generate(ids, C.MAX_NEW, sampling,
                                                       cancel_event or threading.Event()))
            checked()
            last = dict(eng.last or {})
            prompt, decode = float(last.get("prompt_ms") or 0), float(last.get("decode_ms") or 0)
            if n <= 8 or not all(math.isfinite(x) and x > 0 for x in (prompt, decode)):
                raise ValueError("incomplete or non-finite calibration sample")
            if int(last.get("reused") or 0) != 0:
                raise ValueError("fresh-prompt calibration reused cached tokens")
            # Early EOS must not win merely by generating a shorter answer.
            normalized = prompt + decode * C.MAX_NEW / n
            readings.append({"kind": kind, "input_tokens": len(ids), "output_tokens": n,
                             "prompt_ms": prompt, "decode_ms": decode,
                             "client_wall_ms": (time.monotonic() - started) * 1000,
                             "normalized_request_ms": normalized, "decode_tok_s": n * 1000 / decode})
        # Equal weight for short and long workloads, irrespective of case count.
        group_costs = [statistics.geometric_mean(r["normalized_request_ms"] for r in readings if r["kind"] == k)
                       for k in ("short", "long")]
        return {"score": 1000 / statistics.geometric_mean(group_costs), "samples": readings}

    def open_profile(prefill, name):
        candidate = list(baseline_args) if name == "baseline" and baseline_args is not None else C.with_arg(tuned, "--prefill", prefill)
        if name != "baseline":
            candidate = C.with_arg(candidate, "--short-read", "64")
        candidate = C.with_arg(candidate, "--prompt-cache", "0")
        candidate = C.with_arg(candidate, "--conversation-cache-mib", "0")
        eng, actual_args, info = start_safe(candidate, name)
        try:
            if not (1 <= int(info["dflash_block"]) <= int(info["dflash_max"]) <= 7 and
                    0 <= float(info["spec_min_p"]) <= 1):
                raise ValueError("invalid loaded draft settings")
            # An empty tune preserves the original policy, including automatic
            # block selection; forcing the reported maximum would change it.
            return eng, {"args": actual_args, "info": info, "tune": {}}
        except BaseException:
            C.close(eng)
            raise

    # Screen prompt allocations first. Each restart remeasures actual residency;
    # no expert count from another card or another prefill allocation is reused.
    values = [(C.arg_value(tuned, "--prefill"), "baseline")]
    values.extend((str(n), f"prefill-{n}") for n in PREFILLS if n <= context)
    for i, (prefill, name) in enumerate(values):
        checked()
        eng = None
        C._progress(progress, "dflash2_profiles", i, len(values))
        if (name != "baseline" and default is not None and
                str(prefill) == C.arg_value(default["args"], "--prefill") and
                int(C.arg_value(default["args"], "--short-read") or 64) == 64):
            continue
        try:
            eng, profile = open_profile(prefill, name)
            sample(eng, profile["tune"])  # page/graph warmup, excluded
            reading = sample(eng, profile["tune"])
            profile["screen"] = reading
            profiles[name] = profile
            if name == "baseline":
                default = profile
            say(f"    {name}: {reading['score']:.3f} balanced requests/s, "
                f"{profile['info']['expert_slots']} expert slots")
        except C.CalibrationCancelled:
            raise
        except Exception as exc:
            failure(name, exc)
        finally:
            if eng is not None:
                C.close(eng)
    C._progress(progress, "dflash2_profiles", len(values), len(values))
    if default is None:
        report["skipped"] = "no complete safe baseline; earlier calibration results retained"
        return {}, report

    # Refine the two best allocation profiles, including the baseline. The grid
    # couples actual forward width with confidence; neither is optimized alone.
    shortlist = sorted(profiles, key=lambda k: profiles[k]["screen"]["score"], reverse=True)[:2]
    if "baseline" not in shortlist:
        shortlist.append("baseline")
    contenders = []
    C._progress(progress, "dflash2_joint_grid", 0, len(shortlist))
    for i, name in enumerate(shortlist):
        checked()
        eng = None
        try:
            profile = profiles[name]
            eng, actual_args, info = start_safe(profile["args"], name + "/grid")
            maximum = int(info["dflash_max"])
            if not 1 <= maximum <= 7:
                raise ValueError("unsupported DFlash2 forward capacity")
            blocks = sorted({min(3, maximum), max(1, maximum - 1), maximum,
                             int(info["dflash_block"])})
            floors = sorted(set(FLOORS) | {float(info["spec_min_p"])})
            sample(eng, profile["tune"])
            grid = []
            for block in blocks:
                for floor in floors:
                    tune = {"dflash_block": block, "spec_min_p": floor}
                    result = sample(eng, tune)
                    grid.append({"tune": tune, **result})
            best = max(grid, key=lambda x: x["score"])
            profile.update(args=actual_args, info=info, grid=grid)
            contenders.append((best["score"], name, best["tune"]))
        except C.CalibrationCancelled:
            raise
        except Exception as exc:
            failure(name + "/grid", exc)
        finally:
            if eng is not None:
                C.close(eng)
        C._progress(progress, "dflash2_joint_grid", i + 1, len(shortlist))
    report["profiles"] = profiles
    if not contenders:
        return {}, report
    _, best_name, best_tune = max(contenders, key=lambda x: x[0])
    candidate = profiles[best_name]
    confirmations = {"baseline": [], "candidate": []}
    failed = False
    C._progress(progress, "dflash2_confirmation", 0, 2 * C.SWEEP_ROUNDS)
    # Reopen both arms in alternating order. Search readings never enter the
    # confirmation, and confirmation includes additional short/long documents.
    for r in range(C.SWEEP_ROUNDS):
        for key in (("baseline", "candidate") if r % 2 == 0 else ("candidate", "baseline")):
            checked()
            eng = None
            profile, tune = (default, default["tune"]) if key == "baseline" else (candidate, best_tune)
            try:
                eng, actual_args, info = start_safe(profile["args"], "confirm/" + key)
                # A restart that changes the memory budget invalidates the pairing.
                if actual_args != profile["args"] or int(info["expert_slots"]) != int(profile["info"]["expert_slots"]):
                    raise ValueError("expert residency changed between search and confirmation")
                sample(eng, tune)
                confirmations[key].append(sample(eng, tune, full=True))
            except C.CalibrationCancelled:
                raise
            except Exception as exc:
                failure(f"confirm/{r}/{key}", exc)
                failed = True
            finally:
                if eng is not None:
                    C.close(eng)
            C._progress(progress, "dflash2_confirmation", 2 * r + (1 if key == ("baseline" if r % 2 == 0 else "candidate") else 2),
                        2 * C.SWEEP_ROUNDS)
        if failed:
            break
    report["confirmation"] = confirmations
    gains = [c["score"] / b["score"] - 1 for b, c in zip(confirmations["baseline"], confirmations["candidate"])]
    keep = not failed and len(gains) == C.SWEEP_ROUNDS and min(gains) > 0 and statistics.median(gains) > C.MIN_GAIN
    report["paired_gains"] = gains
    if not keep:
        say("    DFlash2 profile: no confirmed improvement; earlier settings retained")
        return {}, report
    report.update(selected=best_name, selected_tune=best_tune,
                  confirmation_gain=statistics.median(gains))
    # Return the whole measured hardware configuration. In particular, a
    # baseline-derived winner must not inherit the rejected legacy proposal's
    # PCIe share, worker count or adaptive policy when it is saved.
    flags = set(C.DEFAULTS) | {"--prefill", "--short-read", "--vram-reserve-mib", "--expert-cache"}
    delta = {flag: value for flag in sorted(flags)
             if (value := C.arg_value(candidate["args"], flag)) is not None}
    if delta.get("--expert-cache") == "auto":
        del delta["--expert-cache"]  # all profiles preserve this original auto choice
    delta.update({"--dflash-block": str(best_tune["dflash_block"]),
                  "--spec-min-p": str(best_tune["spec_min_p"])})
    say(f"    DFlash2 profile confirmed: {100 * statistics.median(gains):.1f}% faster balanced fresh requests")
    return delta, report
