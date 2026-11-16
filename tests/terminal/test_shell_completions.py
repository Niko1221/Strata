"""`strata complete` prints a completion script generated from parser() itself.

The script lists every subcommand because it is built from the same grammar the CLI runs, so there is
no second, hand-kept shell copy that can drift. These tests keep that true: they regenerate the script
and check it names every current top-level command and parses as valid shell."""
import argparse
import shutil
import subprocess
import sys

import pytest

from strata_cli import cli

SHELLS = ("bash", "zsh", "fish")


def _top_level_commands() -> list[str]:
    sub = [action for action in cli.parser()._actions if isinstance(action, argparse._SubParsersAction)]
    return sorted(sub[0].choices)


@pytest.mark.parametrize("shell", SHELLS)
def test_complete_prints_a_script_naming_every_command(shell, capsys):
    pytest.importorskip("shtab")
    assert cli.main(["complete", "--shell", shell]) == 0
    script = capsys.readouterr().out
    assert script, f"empty {shell} script"
    for command in _top_level_commands():
        assert command in script, f"{shell} script is missing the {command} command"


@pytest.mark.parametrize("shell", SHELLS)
def test_complete_script_is_valid_shell_syntax(shell, capsys, tmp_path):
    pytest.importorskip("shtab")
    binary = shutil.which(shell)
    if not binary:
        pytest.skip(f"no {shell} on this box to check the script's syntax")
    assert cli.main(["complete", "--shell", shell]) == 0
    script_file = tmp_path / f"strata.{shell}"
    script_file.write_text(capsys.readouterr().out, encoding="utf-8")
    check = subprocess.run([binary, "-n", str(script_file)], capture_output=True, text=True)
    assert check.returncode == 0, check.stderr


def test_complete_without_shtab_names_the_package(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "shtab", None)      # import shtab raises ImportError
    assert cli.main(["complete"]) == 2
    assert "shtab" in capsys.readouterr().err
