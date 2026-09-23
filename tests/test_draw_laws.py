"""``draw_laws``: counter-based draws of a model's effects, read a tile of rows at a time.

The mixture law's own moments are checked on its draws [own-sim: math only]; the generator against Philox4x32-10's
published known-answer vectors."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import stats

from sv_pgs.draw_laws import COMPONENT_STREAM, DenseDraws, ProductMixtureDraws, export_draws, philox4x32, tile_rows
from sv_pgs.scale_mixture_ep import _components


def test_philox_matches_the_random123_known_answers():
    # Random123 kat_vectors, philox4x32 with 10 rounds: (counter, key) -> output.
    cases = [
        ((0, 0, 0, 0), (0, 0), (0x6627E8D5, 0xE169C58D, 0xBC57AC4C, 0x9B00DBD8)),
        ((0xFFFFFFFF,) * 4, (0xFFFFFFFF, 0xFFFFFFFF), (0x408F276D, 0x41C83B0E, 0xA20BC7C6, 0x6D5451FD)),
        ((0x243F6A88, 0x85A308D3, 0x13198A2E, 0x03707344), (0xA4093822, 0x299F31D0), (0xD16CFE09, 0x94FDCCEB, 0x5001E420, 0x24126EA1)),
    ]
    for counter, key, expected in cases:
        words = philox4x32(tuple(np.array([value], dtype=np.uint64) for value in counter), key)
        assert tuple(int(word[0]) for word in words) == expected


def _law(generator: np.random.Generator, rows: int = 300, draws: int = 4000, weights=(0.6, 0.4)) -> ProductMixtureDraws:
    grid = np.linspace(-7.0, 1.0, 9)
    density = generator.normal(size=(2, grid.shape[0]))
    density -= np.log(np.sum(np.exp(density), axis=1, keepdims=True))
    return ProductMixtureDraws(
        class_index=generator.integers(0, 2, rows), log_scale=generator.normal(-1.0, 0.3, rows),
        omega=generator.uniform(20.0, 200.0, rows)[None, :] / generator.uniform(0.5, 1.5, len(weights))[:, None],
        log_density=density, log_variance_grid=grid, shift=generator.normal(0.0, 3.0, size=(len(weights), rows)),
        weights=np.asarray(weights), key=(17, 29), draw_count=draws,
    )


def test_a_tile_draws_the_same_numbers_however_the_rows_are_split_and_a_subset_draws_its_rows_own():
    law = _law(np.random.default_rng(1), rows=97, draws=13)
    whole = law.tile(0, 97)
    pieces = np.concatenate([law.tile(start, min(start + 10, 97)) for start in range(0, 97, 10)])
    np.testing.assert_array_equal(whole, pieces)
    chosen = np.array([3, 40, 41, 96])
    np.testing.assert_array_equal(law.subset(chosen).tile(0, 4), whole[chosen])
    np.testing.assert_array_equal(np.asarray(law), whole)
    assert tile_rows(law, 1) == 1 and tile_rows(DenseDraws(whole), 8 * 13 * 5) == 5


def test_the_mixture_draws_have_each_components_moments_and_components_follow_the_weights():
    """Each draw is a mixture sample: its component ~ Categorical(w), then every member from that component's q_cj. Per
    component, each member's draws have q_cj's mean and variance within Monte Carlo error [own-sim]."""
    law = _law(np.random.default_rng(2))
    draws = law.tile(0, law.shape[0])
    components = law.components_of_draws()
    counts = np.bincount(components, minlength=2)
    assert stats.binomtest(int(counts[0]), law.draw_count, 0.6).pvalue > 1e-3
    classes = np.asarray(law.class_index)
    for component in range(2):
        columns = np.flatnonzero(components == component)
        values = draws[:, columns]
        omega = law.omega[component]
        mean, variance, fourth = (np.empty(law.shape[0]) for _ in range(3))
        for class_position in range(2):
            members = np.flatnonzero(classes == class_position)
            terms = _components(law.log_density[class_position], law.log_scale[members], law.log_variance_grid, omega[members], law.shift[component, members])
            weight, conditional = terms.responsibility, terms.conditional_variance
            node_means = law.shift[component, members][:, None] * conditional
            mean[members] = np.sum(weight * node_means, axis=1)
            offset = node_means - mean[members][:, None]
            variance[members] = np.sum(weight * (conditional + offset**2), axis=1)
            fourth[members] = np.sum(weight * (offset**4 + 6.0 * offset**2 * conditional + 3.0 * conditional**2), axis=1)
        count = columns.shape[0]
        # Five standard errors of the sample mean and of the sample variance (whose variance is (mu_4 - v^2) / N).
        assert np.all(np.abs(values.mean(axis=1) - mean) <= 5.0 * np.sqrt(variance / count))
        assert np.all(np.abs(values.var(axis=1) - variance) <= 5.0 * np.sqrt((fourth - variance**2) / count) + variance / count)


def test_the_component_counter_stream_is_its_own():
    # The component uniforms use counter (0, 0, k, COMPONENT_STREAM): never a member's (row, draw, MEMBER_STREAM).
    assert COMPONENT_STREAM != 0
    law = _law(np.random.default_rng(3), rows=5, draws=64, weights=(1.0,))
    assert np.all(law.components_of_draws() == 0)


def test_an_export_streams_the_draws_to_disk_a_tile_at_a_time(tmp_path: Path):
    law = _law(np.random.default_rng(4), rows=53, draws=7)
    export_draws(law, tmp_path / "draws.npy", working_bytes=law.tile_row_bytes() * 5)
    np.testing.assert_array_equal(np.load(tmp_path / "draws.npy"), law.tile(0, 53))
