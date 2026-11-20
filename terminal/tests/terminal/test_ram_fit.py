"""tests/terminal/test_ram_fit.py - the RAM-side helpers of a start on a small PC (strata_control/ram_fit.py).

The commit-room math mirrors the engine's clamp (src/core/expert_source.cpp, clamp_resident_budget): these tests pin
the numbers so the two cannot drift apart silently.  The squeeze is exercised only through its test fake
(STRATA_FAKE_SQUEEZE) and its off/auto gates: nothing here touches another program's memory."""
from __future__ import annotations

from pathlib import Path

from strata_control import ram_fit

ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------- the commit room (the engine's clamp, mirrored)


def test_commit_room_mirrors_the_engine_clamp():
    """20 GiB of commit free: the engine's own pre-copy commit (2.0), the copy headroom (4) and the clamp margin
    (0.25) come off first - 20 - 2 - 4 - 0.25 = 13.75."""
    assert ram_fit.commit_room_gib(20.0) == 13.75


def test_commit_room_follows_the_headroom_env():
    import os
    os.environ["STRATA_RESIDENT_HEADROOM_GIB"] = "2"
    try:
        assert ram_fit.commit_room_gib(20.0) == 15.75
    finally:
        del os.environ["STRATA_RESIDENT_HEADROOM_GIB"]


def test_commit_room_rounds_down_to_a_quarter():
    """13.9 - 6.25 = 7.65, offered as 7.5: the engine grants a little less than the estimate, never more."""
    assert ram_fit.commit_room_gib(13.9) == 7.5


def test_commit_room_is_none_without_the_commit_limit():
    """Linux reads no commit limit: no estimate, no advice."""
    assert ram_fit.commit_room_gib(None) is None


def test_commit_room_floors_at_zero():
    assert ram_fit.commit_room_gib(1.0) == 0.0


# ---------------------------------------------------------------- the cut notes


def test_cut_notes_say_what_gives_the_room_back(monkeypatch):
    """A 24 GiB budget with 12 GiB of commit free (room 5.75): two lines - the cut with the numbers, and the page
    file plus the programs that hold the commit."""
    mem = {"total_gib": 31.6, "avail_gib": 20.0, "commit_free_gib": 12.0}
    monkeypatch.setattr(ram_fit, "top_commit_apps", lambda: [("chrome", 4.2)])
    lines = ram_fit.commit_cut_notes(24.0, mem)
    assert len(lines) == 2
    assert "commit limit" in lines[0] and "12" in lines[0] and "5.75" in lines[0]
    assert "system managed" in lines[1] and "chrome 4.2 GiB" in lines[1]


def test_cut_notes_stay_quiet_when_the_budget_fits():
    mem = {"total_gib": 64.0, "avail_gib": 50.0, "commit_free_gib": 60.0}
    assert ram_fit.commit_cut_notes(40.0, mem) == []


def test_cut_notes_without_the_commit_limit():
    mem = {"total_gib": 32.0, "avail_gib": 20.0, "commit_free_gib": None}
    assert ram_fit.commit_cut_notes(40.0, mem) == []


def test_cut_notes_survive_an_unreadable_machine(monkeypatch):
    monkeypatch.setattr(ram_fit, "mem_gib", lambda: None)
    assert ram_fit.commit_cut_notes(40.0) == []


# ---------------------------------------------------------------- the budget in a start's arguments


def test_budget_from_args():
    assert ram_fit.budget_from_args(["--model", "m", "--resident-budget-gib", "38"]) == 38.0
    assert ram_fit.budget_from_args(["--model", "m"]) is None
    assert ram_fit.budget_from_args(["--resident-budget-gib"]) is None
    assert ram_fit.budget_from_args(["--resident-budget-gib", "abc"]) is None


# ---------------------------------------------------------------- the squeeze gates (test fake only)


def test_squeeze_fake_runs_and_reports(monkeypatch):
    monkeypatch.setenv("STRATA_FAKE_SQUEEZE", "10.0:14.5")
    monkeypatch.delenv("STRATA_RAM_SQUEEZE", raising=False)
    out = ram_fit.squeeze(budget_gib=38.0)
    assert out["ran"] is True
    assert out["freed_gib"] == 4.5
    assert "test fake" in out["detail"]


def test_squeeze_off_never_runs(monkeypatch):
    monkeypatch.setenv("STRATA_FAKE_SQUEEZE", "10.0:14.5")
    out = ram_fit.squeeze(mode="off", budget_gib=38.0)
    assert out["ran"] is False and out["detail"] == ""


def test_squeeze_auto_needs_a_budget(monkeypatch):
    """auto (the default) squeezes only where the start asks for a RAM budget: every other start stays as it was."""
    monkeypatch.setenv("STRATA_FAKE_SQUEEZE", "10.0:14.5")
    monkeypatch.delenv("STRATA_RAM_SQUEEZE", raising=False)
    assert ram_fit.squeeze(budget_gib=None)["ran"] is False
    assert ram_fit.squeeze(budget_gib=38.0)["ran"] is True


def test_squeeze_on_forces_without_a_budget(monkeypatch):
    monkeypatch.setenv("STRATA_FAKE_SQUEEZE", "10.0:14.5")
    assert ram_fit.squeeze(mode="on", budget_gib=None)["ran"] is True


def test_squeeze_env_decides(monkeypatch):
    monkeypatch.setenv("STRATA_FAKE_SQUEEZE", "10.0:14.5")
    monkeypatch.setenv("STRATA_RAM_SQUEEZE", "off")
    assert ram_fit.squeeze(budget_gib=38.0)["ran"] is False
    monkeypatch.setenv("STRATA_RAM_SQUEEZE", "on")
    assert ram_fit.squeeze(budget_gib=None)["ran"] is True


def test_squeeze_stays_quiet_when_nothing_came_out(monkeypatch):
    """A squeeze that freed nothing prints nothing: starts stay readable."""
    monkeypatch.setenv("STRATA_FAKE_SQUEEZE", "14.5:14.5")
    out = ram_fit.squeeze(budget_gib=38.0)
    assert out["ran"] is True and out["detail"] == "" and out["freed_gib"] == 0.0


def test_squeeze_bad_fake_is_ignored(monkeypatch):
    monkeypatch.setenv("STRATA_FAKE_SQUEEZE", "not-a-number")
    out = ram_fit.squeeze(budget_gib=38.0)
    assert out["ran"] is False


# ---------------------------------------------------------------- the wiring


def test_the_server_squeezes_before_the_model_loads_and_warns():
    """The squeeze line sits between the start's cards and the model load, and the budget warning reads the same
    arguments after it (the freed RAM is what the warning then measures)."""
    src = (ROOT / "serve" / "server.py").read_text(encoding="utf-8")
    sq = src.index("ram_fit.squeeze(")
    load = src.index("loading the model (the first start takes a minute or two)")
    warn = src.index("warn_budget_over_ram(budget_args)")
    assert sq < load < warn


def test_setup_says_when_the_commit_limit_cuts_the_budget():
    src = (ROOT / "setup.py").read_text(encoding="utf-8")
    assert "ram_fit.commit_cut_notes(budget)" in src
