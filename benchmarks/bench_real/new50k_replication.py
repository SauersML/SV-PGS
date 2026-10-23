"""Workspace-only orchestration of the fixed new-50k replications.

Loading outcomes or scoring is gated on a nonempty, one-to-one workspace map.
Saved LDpred2 Units are located by reproducing their original cache identity,
including record order, code, training rows, tuning split and covariates. Nothing
is fitted. Configuration paths alone are remapped to the app's read-only mounts.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re

import numpy as np

OPS = Path('/home/jupyter/workspace/svpgs_ops/svpgs')
WORK = Path('/home/jupyter/x50k/codex50k')


def base_name(name):
    for script in sorted((OPS / name / 'launcher').glob('*/run_step.sh'), reverse=True):
        match = re.search(r'--base "gs://[^\"]*/svpgs/([^/\"]+)/run"', script.read_text())
        if match and match[1] != name:
            return match[1]
    return None


def remap(value):
    prefixes = {
        '/mnt/disks/ops': str(OPS.parent),
        '/mnt/disks/cdr': '/home/jupyter/workspace/vwb-aou-datasets-controlled/v9',
        '/mnt/disks/imputed': str(WORK / 'mount_imputed'),
        '/mnt/disks/panel': str(WORK / 'mount_panel'),
    }
    if isinstance(value, str):
        for before, after in prefixes.items():
            if value == before or value.startswith(before + '/'):
                return after + value[len(before):]
    if isinstance(value, dict):
        return {key: remap(item) for key, item in value.items()}
    if isinstance(value, list):
        return [remap(item) for item in value]
    return value


def load_run(name):
    from sv_pgs.compute_budget import detect_compute_budget
    from sv_pgs.workspace_pipeline import WorkspaceConfig, _Run, workspace_bindings
    root = OPS / name
    raw = json.loads((root / 'run_config.json').read_text())
    names, queue = [], [name]
    while queue:
        current = queue.pop(0)
        if current in names:
            continue
        names.append(current)
        config = json.loads((OPS / current / 'run_config.json').read_text())
        base = base_name(current)
        if base:
            queue.append(base)
        source = config.get('accept_steps_from')
        if source:
            queue.append(source.rstrip('/').split('/')[-2])
    view = WORK / 'views' / name
    view.mkdir(parents=True, exist_ok=True)
    # Relative store references (including .bases) retain their original tree.
    for current in names:
        run = OPS / current / 'run'
        if not run.exists():
            continue
        for step in run.iterdir():
            destination = view / step.name
            if destination.exists() or destination.is_symlink():
                continue
            destination.symlink_to(step)
    config = remap(raw)
    config['run_directory'] = str(view)
    config['marker_directory'] = str(WORK / 'markers' / name)
    path = WORK / 'views' / (name + '.json')
    path.write_text(json.dumps(config))
    return _Run(WorkspaceConfig.read(path), workspace_bindings(), detect_compute_budget(), {})


def saved_fast_pair(name, trait, run):
    """Resolve the exact original Units key, never choose among weight files by results."""
    from benchmarks.bench_real import aou_fastpgs as fast
    from benchmarks.bench_real.extra_test_people import fast_method_weights
    from sv_pgs.dosage_store import DosageStore
    from sv_pgs.workspace_pipeline import _fit_cohort, _final_store, _final_measurement_model, fit_filtered

    config = run.config
    cohort = _fit_cohort(run)
    index = cohort.trait_names.index(trait)
    measurement = run.bindings.load_measurement_model(_final_measurement_model(run))
    with DosageStore.open(_final_store(run)) as store:
        table = store.variant_table
        arguments = (table, store.n_samples, measurement.log_reliability,
                     config.fit_minimum_minor_allele_frequency, config.fit_minimum_r2, config.fit_variant_sites)
        region = getattr(config, 'fit_region', None)
        kept = fit_filtered(*arguments, region) if region is not None else fit_filtered(*arguments)
        records, cm, short = fast.arm_records(table, kept)
    del measurement
    source = OPS / name / 'source'
    files = ['benchmarks/bench_real/aou_fastpgs.py', 'benchmarks/bench_real/aou_fastpgs.R',
             'benchmarks/bench_sim/submissions/ldpred2_auto_allvar.py',
             'benchmarks/bench_sim/submissions/ldpred2_auto.py', 'benchmarks/bench_sim/submissions/sbayesrc.py']
    digest = hashlib.sha256()
    for file in files:
        digest.update((source / file).read_bytes())
    identity = {'format': 'aou-fastpgs-v1', 'code': digest.hexdigest(), 'steps': fast.step_keys(run),
                'config': (OPS / name / 'run_config.json').read_text()}
    units = fast.Units(OPS / name / 'fastpgs-units', identity)
    model, training, held = fast.trait_rows(cohort, index)
    tuning = fast.tuning_slice(cohort.families[training], fast._seed(int(config.seed), trait))
    own = np.flatnonzero(cohort.covariate_columns[index])
    extra = {'methods': ['ldpred2_auto']} if name == 'aou-scr-d1' else {}
    path = units.path(trait=trait, model=model, records=fast._digest(records), cm=fast._digest(cm),
                      short=fast._digest(short), training=fast._digest(cohort.store_columns[training]),
                      held=fast._digest(cohort.store_columns[held]), tuning=fast._digest(tuning),
                      target=fast._digest(cohort.targets[training, index]),
                      covariates=fast._digest(cohort.covariates[np.ix_(training, own)]), **extra)
    if not path.is_file():
        raise ValueError('Original weight identity does not resolve; no refitting or weight substitution permitted')
    if name == 'aou-m1p-k' and path.name != '98d472b37b06b9b636de8d58d326fa26257a9e7676cb746e942e36ef69b384c3.npz':
        raise ValueError('CKD weights differ from the fixed pre-specification')
    models = [fast_method_weights(path, records, arm + '|ldpred2_auto|auto') for arm in ('snv_indel', 'with_sv')]
    if any(not np.all(np.isfinite(model.weights)) for model in models):
        raise ValueError('Saved arm has nonfinite coefficients')
    return models, path


def require_map(path):
    from benchmarks.bench_real.extra_test_people import read_crosswalk
    if not Path('/home/jupyter/workspace/vwb-aou-datasets-controlled').is_dir():
        raise RuntimeError('Execution restricted to AoU workspace')
    mapping = read_crosswalk(path)
    if len(mapping) < 40:
        raise ValueError('Insufficient mapped participants; scoring prohibited')
    return mapping
