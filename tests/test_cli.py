from __future__ import annotations

from sv_pgs import cli


def test_main_reports_keyboard_interrupt_without_traceback(monkeypatch, capsys):
    def raise_interrupt(_argv):
        raise KeyboardInterrupt("signal SIGTERM")

    monkeypatch.setattr(cli, "_main_impl", raise_interrupt)

    assert cli.main(["list-all-of-us-diseases"]) == 130

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "[sv-pgs] interrupted: signal SIGTERM\n"
    assert "Traceback" not in captured.err


def test_run_mirror_reads_the_bases_it_is_given_and_those_its_mirror_names(monkeypatch, tmp_path):
    """run-mirror --base: each gs:// mirror is opened as a GcsRemote, here a local directory per URL."""
    from types import SimpleNamespace

    from sv_pgs import run_mirror
    from sv_pgs.workspace_pipeline import COMPLETE_FILE, WorkspaceConfig
    from tests.directory_remote import DirectoryRemote

    opened = {}

    def remote(url):
        opened.setdefault(url, DirectoryRemote(tmp_path / url.removeprefix("gs://")))
        return opened[url]

    base = tmp_path / "ops/svpgs/base/run"
    (base / "store").mkdir(parents=True)
    (base / "store" / COMPLETE_FILE).write_text('{"key": "store"}')
    monkeypatch.setattr(run_mirror, "GcsRemote", remote)
    monkeypatch.setattr(WorkspaceConfig, "read", staticmethod(lambda _path: SimpleNamespace(run_directory=tmp_path / "run")))
    mirror = ["--config", "config.json", "--mirror", "gs://ops/svpgs/run/run"]
    assert cli.main(["run-mirror", "restore", *mirror, "--base", "gs://ops/svpgs/base/run"]) == 0
    assert (tmp_path / "run" / "store" / COMPLETE_FILE).read_text() == '{"key": "store"}'
    assert cli.main(["run-mirror", "push", *mirror, "--base", "gs://ops/svpgs/base/run"]) == 0
    assert sorted(path.relative_to(tmp_path / "ops/svpgs/run/run").as_posix() for path in (tmp_path / "ops/svpgs/run/run").rglob("*") if path.is_file()) == [
        ".bases/ops/svpgs/base/run"
    ]
    # A later job given no base reads the one its mirror names.
    (tmp_path / "run" / "store").rename(tmp_path / "elsewhere")
    assert cli.main(["run-mirror", "restore", *mirror]) == 0
    assert (tmp_path / "run" / "store" / COMPLETE_FILE).read_text() == '{"key": "store"}'
