"""A corrupt / truncated cache file degrades to a clean miss, never a crash.

A crash mid-write (or an interrupted np.savez) leaves a half-written cache on
disk. The next run must treat such a file as absent and recompute, rather than
aborting a multi-hour job with an uncaught zipfile.BadZipFile / UnpicklingError
(neither of which is an OSError/ValueError subclass).
"""
from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np

from sv_pgs import artifact as artifact_mod
from sv_pgs import io as io_mod
from sv_pgs import model as model_mod


def test_corruption_error_sets_include_the_non_os_value_exceptions():
    # The whole point of the shared tuples: catch the exceptions that are NOT
    # OSError/ValueError subclasses, which a naive (OSError, ValueError) misses.
    for errors in (model_mod._CACHE_CORRUPTION_ERRORS, io_mod._CACHE_CORRUPTION_ERRORS):
        assert zipfile.BadZipFile in errors
        assert __import__("pickle").UnpicklingError in errors
    assert zipfile.BadZipFile in artifact_mod._ARTIFACT_CORRUPTION_ERRORS


def test_truncated_npz_raises_badzipfile_and_is_in_the_caught_set(tmp_path: Path):
    # Write a valid .npz, then truncate it to simulate a crash mid-write.
    npz_path = tmp_path / "cache.npz"
    np.savez(npz_path, value=np.arange(1000, dtype=np.int64))
    raw = npz_path.read_bytes()
    npz_path.write_bytes(raw[: len(raw) // 2])

    # np.load on the truncated file raises BadZipFile (NOT OSError/ValueError) —
    # this is exactly the exception our cache loaders must catch.
    try:
        with np.load(npz_path) as handle:
            _ = handle["value"]
        raised: BaseException | None = None
    except BaseException as exc:  # noqa: BLE001 — we assert on the type below
        raised = exc

    assert raised is not None
    assert isinstance(raised, zipfile.BadZipFile)
    assert isinstance(raised, model_mod._CACHE_CORRUPTION_ERRORS)
    assert isinstance(raised, io_mod._CACHE_CORRUPTION_ERRORS)


def test_try_load_artifact_returns_none_on_truncated_npz(tmp_path: Path):
    # A results dir whose arrays.npz is truncated must read as a clean miss
    # (refit), not crash the reuse check.
    (tmp_path / "metadata.json").write_text('{"fit_fingerprint": "abc"}', encoding="utf-8")
    npz_path = tmp_path / "arrays.npz"
    np.savez(npz_path, tie_kept_indices=np.arange(10, dtype=np.int32))
    raw = npz_path.read_bytes()
    npz_path.write_bytes(raw[: len(raw) // 2])  # truncate

    result = artifact_mod.try_load_artifact_if_fingerprint_matches(tmp_path, "abc")
    assert result is None
