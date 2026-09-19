import subprocess
import sys
from importlib.metadata import PackageNotFoundError

import pytest

from sv_pgs import cli


def test_version_runs():
    completed = subprocess.run([sys.executable, "-m", "sv_pgs.cli", "version"], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
    assert "sv-pgs" in completed.stdout


def test_version_is_unknown_only_when_the_package_is_not_installed(monkeypatch):
    def not_installed(_name):
        raise PackageNotFoundError("sv-pgs")

    monkeypatch.setattr(cli, "package_version", not_installed)
    assert cli._resolve_version_info()[0] == "unknown"

    def broken_metadata(_name):
        raise ValueError("corrupt metadata")

    monkeypatch.setattr(cli, "package_version", broken_metadata)
    with pytest.raises(ValueError, match="corrupt metadata"):
        cli._resolve_version_info()


def test_commit_is_unknown_when_git_is_missing(monkeypatch):
    def missing_git(*_args, **_kwargs):
        raise FileNotFoundError("git")

    monkeypatch.setattr(cli.subprocess, "run", missing_git)
    assert cli._resolve_version_info()[1] == "unknown"
