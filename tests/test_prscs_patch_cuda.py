"""The patched PRS-CS block preparation on the GPU: when the blocks held on the device leave too little for the next
block's eigendecomposition, they move to the host and the preparation still matches PRS-CS's own (CPU SVD) result."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from sv_pgs.compute_budget import _try_import_cupy

cupy = _try_import_cupy()
pytestmark = pytest.mark.skipif(cupy is None, reason="needs a CUDA device")
pytest.importorskip("h5py", reason="PRS-CS's parse_genet reads its reference with h5py")

PATCH = Path(__file__).resolve().parents[1] / "benchmarks" / "bench_sim" / "prscs_patch" / "parse_genet.py"


def _parse_genet():
    spec = importlib.util.spec_from_file_location("prscs_parse_genet", PATCH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _inputs(rng: np.random.Generator):
    sizes = (30, 45, 25)
    blocks, snps = [], []
    for number, size in enumerate(sizes):
        x = rng.standard_normal((size, 4 * size))
        correlation = np.corrcoef(x)
        correlation[0, 1] = correlation[1, 0] = correlation[0, 1] + 1e-3  # a slightly indefinite block, as LD can be
        blocks.append(correlation)
        snps.append([f"b{number}_{index}" for index in range(size)])
    names = [name for block in snps for name in block]
    flips = rng.choice([-1.0, 1.0], size=len(names)).tolist()
    return blocks, snps, {"SNP": names, "FLP": flips}


def test_the_blocks_move_to_the_host_when_the_device_runs_short(monkeypatch) -> None:
    parse_genet = _parse_genet()
    blocks, snps, sst = _inputs(np.random.default_rng(0))
    monkeypatch.delenv("PRSCS_DEVICE", raising=False)
    expected, expected_sizes = parse_genet.prepare_ldblk([b.copy() for b in blocks], snps, sst)

    monkeypatch.setenv("PRSCS_DEVICE", "gpu")
    symmetrized = parse_genet._symmetrized
    calls = []

    def short_on_the_third(cp, host, flip):
        calls.append(host.shape[0])
        if len(calls) == 3:
            raise cp.cuda.memory.OutOfMemoryError(host.nbytes, 0, 0)
        return symmetrized(cp, host, flip)

    monkeypatch.setattr(parse_genet, "_symmetrized", short_on_the_third)
    prepared, sizes = parse_genet.prepare_ldblk([b.copy() for b in blocks], snps, sst)
    assert sizes == expected_sizes
    assert calls == [30, 45, 25, 25]
    assert all(isinstance(block, np.ndarray) for block in prepared)
    for got, want in zip(prepared, expected):
        np.testing.assert_allclose(got, want, rtol=0, atol=1e-12)


def test_the_blocks_stay_on_the_device_when_they_fit(monkeypatch) -> None:
    parse_genet = _parse_genet()
    blocks, snps, sst = _inputs(np.random.default_rng(1))
    monkeypatch.delenv("PRSCS_DEVICE", raising=False)
    expected, _ = parse_genet.prepare_ldblk([b.copy() for b in blocks], snps, sst)
    monkeypatch.setenv("PRSCS_DEVICE", "gpu")
    prepared, _ = parse_genet.prepare_ldblk([b.copy() for b in blocks], snps, sst)
    assert all(isinstance(block, cupy.ndarray) for block in prepared)
    for got, want in zip(prepared, expected):
        np.testing.assert_allclose(cupy.asnumpy(got), want, rtol=0, atol=1e-12)
