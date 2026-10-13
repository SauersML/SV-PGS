"""Reject inputs that would silently change a model's likelihood or study folds."""
from types import SimpleNamespace

import numpy as np
import pytest

from sv_pgs.config import TraitType
from sv_pgs.fit_model import FitRequest
from sv_pgs.workspace_pipeline import WorkspaceConfig
from tests.test_workspace_pipeline import BUDGET, workspace


def _request(tmp_path, **overrides):
    fields = dict(
        store=SimpleNamespace(n_samples=4, n_variants=2), store_columns=np.arange(4),
        research_ids=("a", "b", "c", "d"), covariates=np.zeros((4, 2)),
        covariate_names=("age", "pc1"), covariate_columns=np.ones((1, 2), dtype=bool),
        targets=np.array([[0.], [1.], [0.], [1.]]), training=np.ones((4, 1), dtype=bool),
        model_names=("disease",), trait_types=(TraitType.BINARY,), log_variance_offset=None,
        budget=BUDGET, work_dir=tmp_path, seed=0,
    )
    return FitRequest(**(fields | overrides))


@pytest.mark.parametrize("kind", ["binray", "gaussian", "", None, 1])
def test_unknown_trait_type_cannot_silently_select_a_gaussian_likelihood(tmp_path, kind):
    with pytest.raises(ValueError, match="trait_types"):
        _request(tmp_path, trait_types=(kind,))


def test_valid_string_trait_type_is_normalized(tmp_path):
    assert _request(tmp_path, trait_types=("binary",)).trait_types == (TraitType.BINARY,)


def test_duplicate_covariate_names_are_ambiguous_at_scoring(tmp_path):
    with pytest.raises(ValueError, match="covariate_names must be distinct"):
        _request(tmp_path, covariate_names=("age", "age"))


def test_negative_fit_seed_is_refused_before_reading_the_store(tmp_path):
    with pytest.raises(ValueError, match="seed must be non-negative"):
        _request(tmp_path, seed=-1)


@pytest.mark.parametrize("key,value", [
    ("fold_count", 1), ("fold_count", 0), ("fold_count", 2.5), ("fold_count", True), ("fold_count", "5"),
    ("seed", -1), ("seed", 1.5), ("seed", False), ("seed", "0"),
])
def test_workspace_does_not_coerce_invalid_fold_counts_or_seeds(workspace, key, value):
    synthetic, _client = workspace
    with pytest.raises(ValueError, match=key):
        WorkspaceConfig.from_mapping(synthetic.raw | {key: value})


def test_chromosome_aliases_cannot_duplicate_a_chromosome(workspace):
    synthetic, _client = workspace
    with pytest.raises(ValueError, match="each store chromosome once"):
        WorkspaceConfig.from_mapping(synthetic.raw | {"chromosomes": ["chr21", "chr021"]})
