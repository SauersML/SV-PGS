"""Executable regressions for the October readiness review; all inputs are synthetic."""
from pathlib import Path
from types import SimpleNamespace
import os

import numpy as np

from sv_pgs.content_identity import file_digest, tree_digest
from sv_pgs.draw_laws import DenseDraws
from sv_pgs.workspace_pipeline import genotype_file_fingerprint, variant_effects


def test_middle_edit_invalidates_fingerprint_even_when_size_and_mtime_are_restored(tmp_path):
    path = tmp_path / 'genotypes.bcf'
    path.write_bytes(b'a' * 200_000)
    stamp = path.stat()
    before = genotype_file_fingerprint(path)
    with path.open('r+b') as handle:
        handle.seek(100_000)
        handle.write(b'b')
    os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
    assert genotype_file_fingerprint(path) != before
    assert file_digest(path) == genotype_file_fingerprint(path)[1]


def test_store_digest_binds_shard_interiors_and_annotations(tmp_path):
    (tmp_path / 'MANIFEST.json').write_text('{}')
    shard = tmp_path / 'shard'
    shard.write_bytes(b'aaaa')
    before = tree_digest(tmp_path)
    shard.write_bytes(b'abaa')
    assert tree_digest(tmp_path) != before
    before = tree_digest(tmp_path)
    (tmp_path / 'annotation').write_bytes(b'1')
    assert tree_digest(tmp_path) != before


def test_effect_uncertainty_is_centered_on_the_known_mean():
    draws = np.random.default_rng(1).normal(1, .01, (1, 64))
    model = SimpleNamespace(store_rows=np.array([0]), signed_scales=np.array([127.]),
                            coefficients=np.array([1.]), posterior_draws=DenseDraws(draws))
    effects = variant_effects(model, np.array([127]), 1 << 20)
    expected = np.sqrt(np.mean((draws - 1) ** 2))
    assert expected > 0
    np.testing.assert_allclose(effects.standard_deviation, [expected])


def test_exact_feature_identity_disambiguates_equal_length_and_symbolic_alleles():
    from sv_pgs.feature_identity import chromosome_identity
    args = dict(genome_build='GRCh38', chromosome='chr1', positions=[101], refs=['A'], alts=['C'],
                identifiers=['v1'], ends=[None], sv_types=[None], sv_lengths=[np.nan], codes_per_unit=[127], value_origin=[0])
    reference = chromosome_identity(**args)
    for change in (dict(alts=['G']), dict(refs=['C'], alts=['A']), dict(genome_build='GRCh37'),
                   dict(codes_per_unit=[100]), dict(value_origin=[1]), dict(identifiers=['v2'])):
        assert chromosome_identity(**(args | change)) != reference
    sv = args | dict(alts=['<DEL>'], ends=[501], sv_types=['DEL'], sv_lengths=[400])
    assert chromosome_identity(**sv) != chromosome_identity(**(sv | dict(ends=[601])))


def test_acceptance_quarantines_unresolved_fits_but_allows_null_notes():
    from sv_pgs.model_acceptance import acceptance, require_accepted
    import pytest
    model = SimpleNamespace(model_names=('null',), scoring=(), refusals=('no variable genetic columns',),
                            certificate=dict(outer_criterion_met=np.array([True]), unresolved=np.array([0]),
                                             budget_unresolved=np.array([0]), restored_best=np.array([False]), far_field_scale=np.array([0.])))
    assert acceptance(model)['accepted']
    assert not acceptance(model)['deployment_validated']
    for term, value in (('outer_criterion_met', False), ('budget_unresolved', 1), ('restored_best', True)):
        bad = SimpleNamespace(**(vars(model) | {'certificate': model.certificate | {term: np.array([value])}}))
        with pytest.raises(ValueError, match='quarantined'):
            require_accepted(bad)
    model.certificate['unresolved'] = np.array([3])
    assert acceptance(model)['accepted']  # Failed trials do not invalidate a converged final state.
    model.refusals = ('lattice check unresolved: refit failed',)
    assert not acceptance(model)['accepted']


def test_frozen_calibration_cannot_hide_a_reversed_evaluation_predictor():
    from sv_pgs.probability_calibration import FrozenCalibration, paired_frozen_gain
    from scipy.special import expit
    rng = np.random.default_rng(4)
    scores = rng.normal(size=4000)
    y = rng.binomial(1, expit(scores))
    calibrator = FrozenCalibration.fit(scores[:2000], y[:2000])
    baseline = calibrator.logits(scores[2000:])
    reversed_score = calibrator.logits(-scores[2000:])
    assert paired_frozen_gain(y[2000:], baseline, reversed_score).estimate < -.2
    assert paired_frozen_gain(y[2000:], baseline, baseline).estimate == 0


def test_portable_layout_accepts_new_people_but_refuses_wrong_equal_length_allele(tmp_path):
    import json
    import pytest
    from sv_pgs.feature_identity import IDENTITY_FILE, chromosome_identity
    from sv_pgs.artifact import predict
    from sv_pgs.dosage_store import DosageStore
    from tests.test_dosage_store import _write_store
    from tests.test_artifact import _model, _budget
    first, second = tmp_path / 'train', tmp_path / 'new_people'
    rng = np.random.default_rng(44)
    for root in (first, second):
        _write_store(root, [{'chr22': rng.integers(0, 2001, size=(30, 40))}])
    with DosageStore.open(first) as store:
        positions = store.variant_table.position.tolist()
    args = dict(genome_build='GRCh38', chromosome='chr22', positions=positions, refs=['A'] * 30,
                alts=['C'] * 30, identifiers=[str(i) for i in range(30)], ends=[None] * 30,
                sv_types=[None] * 30, sv_lengths=[np.nan] * 30, codes_per_unit=[127] * 30, value_origin=[0] * 30)
    for root in (first, second):
        (root / IDENTITY_FILE).write_text(json.dumps({'chr22': chromosome_identity(**args)}))
    model = _model(rng, first)
    with DosageStore.open(second) as store:
        assert np.isfinite(predict(model, store, np.arange(4), np.zeros((4, 2)), _budget()).linear_predictor).all()
    args['alts'][0] = 'G'
    (second / IDENTITY_FILE).write_text(json.dumps({'chr22': chromosome_identity(**args)}))
    with DosageStore.open(second) as store, pytest.raises(ValueError, match='variant layout'):
        predict(model, store, np.arange(4), np.zeros((4, 2)), _budget())


def test_mixture_effect_summaries_include_rare_components_without_draw_noise():
    from scipy.special import ndtr
    from sv_pgs.draw_laws import ProductMixtureDraws
    # One node, two tilted normal components with known mean/variance.
    law = ProductMixtureDraws(class_index=np.array([0]), log_scale=np.array([0.]), log_density=np.array([[0.]]),
                             log_variance_grid=np.array([0.]), shift=np.array([[0.], [2.]]), omega=np.array([[1.], [1.]]),
                             weights=np.array([.999, .001]), key=(1, 2), draw_count=1)
    center = np.array([.001])
    variance, positive = law.marginal_summary(0, 1, center)
    np.testing.assert_allclose(variance, [.5 + .999 * .001**2 + .001 * .999**2])
    np.testing.assert_allclose(positive, [.999 * .5 + .001 * ndtr(np.sqrt(2))])


def test_artifact_detects_finite_array_corruption(tmp_path):
    from sv_pgs.artifact import save_model, load_model
    from tests.test_artifact import _model
    from tests.test_dosage_store import _write_store
    import pytest
    rng = np.random.default_rng(22)
    store = tmp_path / 'store'
    _write_store(store, [{'chr22': rng.integers(0, 2001, size=(30, 40))}])
    path = tmp_path / 'model'
    save_model(path, _model(rng, store))
    with np.load(path / 'arrays.npz') as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays['scoring/0/coefficients'][0] += .001
    np.savez(path / 'arrays.npz', **arrays)
    with pytest.raises(ValueError, match='checksum'):
        load_model(path)


def test_a_band_accelerated_full_data_fit_is_accepted_and_only_the_pure_summary_route_is_truncated():
    from sv_pgs.model_acceptance import acceptance
    certificate = dict(outer_criterion_met=np.array([True]), unresolved=np.array([0]), budget_unresolved=np.array([0]),
                       restored_best=np.array([False]))
    # The band-accelerated full-data route records the band's far-field diagnostic and a far-field scale of 0.
    accelerated = SimpleNamespace(model_names=('model',), scoring=(), refusals=(),
                                  certificate=certificate | {'far_field': np.array([0.03]), 'far_field_scale': np.array([0.0])})
    assert acceptance(accelerated)['accepted']
    # The pure summary route left its far field out of the likelihood and records the measured scale.
    summary = SimpleNamespace(**(vars(accelerated) | {'certificate': accelerated.certificate | {'far_field_scale': np.array([1.2])}}))
    assert not acceptance(summary)['accepted'] and 'truncated_ld' in {issue['code'] for issue in acceptance(summary)['issues']}


def test_tree_digest_matches_serial_definition_without_trusting_timestamp_resolution(tmp_path, monkeypatch):
    import hashlib
    import struct
    from sv_pgs import content_identity
    for name, data in (('MANIFEST.json', b'{}'), ('dosage/shard', b'abcd' * 1000), ('annotation', b'1')):
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_bytes(data)
    expected = hashlib.sha256()
    for path in sorted(p for p in tmp_path.rglob('*') if p.is_file()):
        name = path.relative_to(tmp_path).as_posix().encode()
        expected.update(struct.pack('<Q', len(name)) + name + hashlib.sha256(path.read_bytes()).digest())
    before = tree_digest(tmp_path)
    assert before == expected.hexdigest()
    # Simulate coarse/remote metadata with unchanged stamps. Content still determines the key.
    monkeypatch.setattr(content_identity, '_stamp', lambda path: (1, 1, 1, 1, 1))
    assert tree_digest(tmp_path) == before
    (tmp_path / 'annotation').write_bytes(b'2')
    assert tree_digest(tmp_path) != before


def test_exact_zero_draws_are_neither_positive_nor_negative_effects():
    model = SimpleNamespace(store_rows=np.array([0]), signed_scales=np.array([127.]), coefficients=np.array([0.]),
                            posterior_draws=DenseDraws(np.array([[0., 0., -1., 1.]])))
    effects = variant_effects(model, np.array([127]), 1 << 20)
    assert effects.probability_positive[0] == .25
    assert effects.probability_negative[0] == .25
