from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("sv_pgs.gds")

from sv_pgs.gds import cufile_read_to_device, gpudirect_available  # noqa: E402


def test_gpudirect_available_returns_bool() -> None:
    result = gpudirect_available()
    assert isinstance(result, bool)


def test_cufile_read_raises_when_unavailable(tmp_path: Path) -> None:
    if gpudirect_available():
        pytest.skip("GPUDirect Storage available; cannot test unavailable path")

    dummy_path = tmp_path / "dummy.bin"
    dummy_path.write_bytes(b"\x00" * 16)

    with pytest.raises(RuntimeError):
        cufile_read_to_device(dummy_path, object(), 0, 16)
