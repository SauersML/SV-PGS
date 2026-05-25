from __future__ import annotations

from sv_pgs import cli


def test_main_reports_keyboard_interrupt_without_traceback(monkeypatch, capsys):
    def raise_interrupt(_argv):
        raise KeyboardInterrupt("signal SIGTERM")

    monkeypatch.setattr(cli, "_main_impl", raise_interrupt)

    assert cli.main(["run-all-of-us", "--disease", "hypertension", "--output-dir", "out"]) == 130

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == "[sv-pgs] interrupted: signal SIGTERM\n"
    assert "Traceback" not in captured.err
