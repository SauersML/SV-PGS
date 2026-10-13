"""Full-precision operator passes prepare a sample operand once per device."""
import numpy as np
import pytest

from sv_pgs import dual_solve
from tests.test_dual_solve import _coded_problem, _CodeTileSource
from sv_pgs.code_products import CodeBlockTile


@pytest.mark.parametrize("relative_error", [0., 1e-5])
def test_operator_prepares_once_and_matches_unprepared_cpu_products(monkeypatch, relative_error):
    _, codes, means, scales, bounds, covariates, weights, variances, _, response = _coded_problem(28)
    source = dual_solve.StreamedDualSource(_CodeTileSource(codes, means, scales, bounds, np, 1 << 24))
    models = dual_solve.DualModels(weights, variances, covariates)
    model_columns = np.arange(models.model_count)
    left = models.sample_to_design(response, model_columns)
    original = np.zeros_like(response)
    for start, stop, tile in source.blocks():
        original += tile.matmat(variances[start:stop] * tile.rmatmat(left))
    expected = response + models.design_to_sample(original, model_columns)
    prepared = []
    method = CodeBlockTile.sample_operand

    def record(tile, values, error):
        prepared.append(error)
        return method(tile, values, error)

    monkeypatch.setattr(CodeBlockTile, "sample_operand", record)
    result = dual_solve.apply_operator(source, models, response, model_columns, relative_error, dual_solve.PassCount(), "test")
    assert prepared == [max(relative_error, np.finfo(float).eps / 2)]
    # Same host DGEMMs and summation order: preparation reuses their input only.
    np.testing.assert_array_equal(result, expected)
