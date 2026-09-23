"""Genome-scale EP's pieces (``full_data_fit`` EP route): the leave-block-out windows cut to the fit's budget."""

from __future__ import annotations

import numpy as np

from sv_pgs.marginal_variances import BlockGrams, refined_grams, window_width, window_working_bytes


def _banded(seed: int, sizes=(15, 25, 9)) -> tuple[np.ndarray, BlockGrams]:
    generator = np.random.default_rng(seed)
    count = int(sum(sizes))
    design = generator.standard_normal((120, count))
    design[:, 1:] += 0.7 * design[:, :-1]
    starts = np.concatenate([[0], np.cumsum(sizes)])
    blocks = tuple(np.arange(starts[index], starts[index + 1]) for index in range(len(sizes)))
    gram = design.T @ design
    within = tuple(gram[np.ix_(block, block)].astype(np.float32) for block in blocks)
    cross = tuple(gram[np.ix_(blocks[index], blocks[index + 1])].astype(np.float32) for index in range(len(sizes) - 1))
    return gram, BlockGrams(blocks=blocks, within=within, next_cross=cross, scale=0.5)


def test_refined_grams_are_slices_of_the_stored_grams() -> None:
    gram, grams = _banded(1)
    refined = refined_grams(grams, 4)
    assert np.array_equal(np.concatenate(refined.blocks), np.concatenate(grams.blocks))
    assert max(block.shape[0] for block in refined.blocks) <= 4
    assert len(refined.next_cross) == len(refined.blocks) - 1
    for index, block in enumerate(refined.blocks):
        np.testing.assert_allclose(refined.within_block(index), 0.5 * gram[np.ix_(block, block)], rtol=1e-6)
        if index + 1 < len(refined.blocks):
            np.testing.assert_allclose(refined.cross_block(index), 0.5 * gram[np.ix_(block, refined.blocks[index + 1])], rtol=1e-6)
    # Views of the stored arrays, not copies.
    assert np.shares_memory(refined.within[0], grams.within[0])


def test_window_width_keeps_every_window_within_the_budget() -> None:
    _gram, grams = _banded(2, sizes=(400, 300))
    for budget in (10_000, 1 << 20, 1 << 24):
        width = window_width(budget)
        refined = refined_grams(grams, width)
        assert window_working_bytes(refined) <= budget or width == 1
        # The next width up would not fit a window of three blocks at that width.
        assert window_working_bytes(refined_grams(BlockGrams(
            blocks=(np.arange(3 * (width + 1)),), within=(np.zeros((3 * (width + 1),) * 2),), next_cross=(),
        ), width + 1)) > budget
