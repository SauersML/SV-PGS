"""benchmarks/power/imputation_accuracy.py on a small synthetic cohort directory."""
from __future__ import annotations

import numpy as np

from benchmarks.power import imputation_accuracy as accuracy


def test_exact_dosages_score_one_and_noisy_dosages_score_their_squared_correlation(tmp_path) -> None:
    generator = np.random.default_rng(3)
    records, people = 40, 4000
    frequency = np.linspace(0.02, 0.4, records)
    genotypes = generator.binomial(2, frequency[:, None], size=(records, people)).astype(np.uint8)
    classes = np.where(np.arange(records) % 2 == 0, 3, 0)
    noisy = genotypes.astype(float) + np.where(classes[:, None] == 3, generator.normal(0.0, 0.6, (records, people)), 0.0)
    observed = np.clip(np.rint(noisy * 127), 0, 254).astype(np.uint8)
    np.save(tmp_path / "truth_G.npy", genotypes)
    np.save(tmp_path / "observed_beagle.npy", observed)
    np.save(tmp_path / "measured.npy", np.ones(records, dtype=bool))
    np.savez(tmp_path / "variants.npz", cls=classes)
    frequencies, r2 = accuracy.record_r2(tmp_path, np.flatnonzero(classes == 0), "beagle")
    np.testing.assert_allclose(r2, 1.0, atol=1e-12)
    _frequencies, sv_r2 = accuracy.record_r2(tmp_path, np.flatnonzero(classes == 3), "beagle")
    expected = [np.corrcoef(genotypes[row], observed[row])[0, 1] ** 2 for row in np.flatnonzero(classes == 3)]
    np.testing.assert_allclose(sv_r2, expected, rtol=1e-10)
    assert np.all(sv_r2 < 0.95) and np.all(np.abs(frequencies - frequency[classes == 0]) < 0.03)
    result = accuracy.imputation_accuracy(tmp_path)
    assert result["SV"]["records"] == records // 2 and result["SNV"]["records"] == records // 2
    assert sum(entry["count"] for entry in result["SNV"]["bins"]) == records // 2
