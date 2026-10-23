import numpy as np

from benchmarks.bench_real.genotype_identity import assignments, distribution, exact_top_two, pack, shown


def test_packing_retains_missing_and_partial_final_word():
    g = np.tile(np.array([0, 1, 2, 3], dtype=np.uint8), (3, 9))[:, :35]
    code, valid = pack(g)
    for site in range(g.shape[1]):
        np.testing.assert_array_equal((code[:, site // 32] >> np.uint64(2 * (site % 32))) & np.uint64(3), g[:, site])
        np.testing.assert_array_equal((valid[:, site // 32] >> np.uint64(2 * (site % 32))) & np.uint64(1), g[:, site] != 3)


def test_exact_search_matches_brute_force_with_missingness_and_ties():
    rng = np.random.default_rng(19)
    reference = rng.integers(0, 3, (60, 97), dtype=np.uint8)
    query = reference[:40].copy()
    reference[41] = reference[0]
    query[1, :3] = 3
    reference[2, :4] = 3
    reference[4, :20] = 3
    index, score, overlap = exact_top_two(query, reference, 90)
    for i, row in enumerate(query):
        common = (row != 3) & (reference != 3)
        n = common.sum(axis=1)
        expected = np.where(n >= 90, 1 - ((reference != row) & common).sum(axis=1) / n, -1)
        order = np.argsort(-expected, kind='stable')[:2]
        np.testing.assert_array_equal(index[i], order)
        np.testing.assert_allclose(score[i], expected[order])
        np.testing.assert_array_equal(overlap[i], n[order])


def test_assignments_reject_ties_duplicate_targets_and_missing_runner_up():
    index = np.array([[0, 1], [2, 3], [2, 4], [5, 6], [7, -1], [8, 9]])
    scores = np.array([[1, 1], [1, .7], [.999, .6], [.995, .7], [1, -1], [.98, .6]])
    accepted, ambiguous = assignments(index, scores)
    np.testing.assert_array_equal(accepted, [False, False, False, True, False, False])
    np.testing.assert_array_equal(ambiguous, [True, True, True, False, True, False])
    assert shown(0) == shown(39) == '<40'
    assert distribution(np.ones(39))['distribution'] == 'suppressed'
