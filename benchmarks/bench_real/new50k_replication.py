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


def original_info(path, chromosome, keys):
    """Sample-weighted original INFO/AF and INFO/INFO; no individual rows emitted."""
    import subprocess
    import tempfile
    from sv_pgs.store_converter import refalt_digest
    wanted = {(int(p), str(i)): j for j, (p, i) in enumerate(zip(keys['positions'], keys['identifiers']))}
    out = np.zeros((3, len(wanted)))
    with tempfile.NamedTemporaryFile('w', suffix='.tsv') as regions:
        regions.writelines(f'{chromosome}\t{p}\n' for p in np.unique(keys['positions']))
        regions.flush()
        result = subprocess.run(['bcftools', 'query', '-R', regions.name, '-f',
                                 '%POS\t%INFO/ID\t%REF\t%ALT\t%AF\t%INFO\t%AN\n', str(path)],
                                capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError('Original batch metadata extraction failed')
    for line in result.stdout.splitlines():
        p, identifier, ref, alt, af, info, an = line.split('\t')
        j = wanted.get((int(p), identifier))
        if j is None:
            continue
        if refalt_digest(ref, alt) != int(keys['digests'][j]):
            raise ValueError('Original site alleles differ from fixed store')
        if '.' in (af, info, an):
            continue
        frequency, quality, alleles = float(af), float(info), float(an)
        if alleles < 80 or not np.all(np.isfinite([frequency, quality, alleles])):
            continue
        if out[2, j] != 0:
            raise ValueError('Repeated original site key')
        out[:, j] = [frequency * alleles, quality * alleles, alleles]
    return out


def original_drift(run, table):
    from concurrent.futures import ThreadPoolExecutor
    from benchmarks.bench_real import extra_test_people as extra
    from sv_pgs.workspace_pipeline import strata_complexity_legend
    chromosomes, starts, digest = extra.store_layout(run)
    legend = strata_complexity_legend(run.config.popped.strata_directory, chromosomes)
    positions = np.searchsorted(starts, table.rows, side='right') - 1
    sums = np.zeros((3, len(table.rows)))
    for index, chromosome in enumerate(chromosomes):
        members = np.flatnonzero(positions == index)
        if not len(members):
            continue
        keys = extra.chromosome_keys((run.config.popped, chromosome, legend, table.rows[members] - starts[index],
                                      int(starts[index+1] - starts[index]), digest[chromosome]))
        paths = [half.path(chromosome, batch) for half in run.config.imputed_halves for batch in half.batches]
        with ThreadPoolExecutor(max_workers=8) as pool:
            for values in pool.map(lambda path: original_info(path, chromosome, keys), paths):
                sums[:, members] += values
        print('original batch drift metadata ready: ' + chromosome, flush=True)
    with np.errstate(divide='ignore', invalid='ignore'):
        return sums[:2] / sums[2]


def main():
    import argparse
    import os
    import subprocess
    from benchmarks.bench_real import extra_test_people as extra
    from benchmarks.bench_real.sv_gain_replication import batch_drift, drift_lines, replicate, report, shown
    from sv_pgs.dosage_store import DosageStore
    from sv_pgs.workspace_pipeline import _read_cohort, _final_store, structural_records

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--map', required=True, type=Path)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()
    mapping = require_map(args.map)  # Must precede every model/outcome operation.
    os.umask(0o077)
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    os.environ['GCS_OAUTH_TOKEN'] = subprocess.run(['gcloud', 'auth', 'print-access-token'],
                                                  capture_output=True, text=True, check=True).stdout.strip()
    os.environ['GCS_REQUESTER_PAYS_PROJECT'] = os.environ['GOOGLE_PROJECT']
    extra.OPS = str(OPS)
    extra.load_run = load_run
    out = WORK / 'replication'
    out.mkdir(exist_ok=True)
    excluded = frozenset((Path.home() / 'xw/longread_ids.txt').read_text().split())
    specifications = [('chronic_kidney_disease', 'aou-m1p-k'),
                      ('hypothyroidism', 'aou-x3-o1'), ('psoriasis', 'aou-scr-d1')]
    runs, models, entries, layouts, provenance = {}, [], [], [], {}
    for trait, name in specifications:
        run = load_run(name)
        runs[trait] = run
        if trait == 'hypothyroidism':
            pair = [extra.model_weights(extra.model_folder(name, arm, trait)) for arm in ('snv_indel', 'with_sv')]
            provenance[trait] = {'run': name, 'kind': 'fixed SV-PGS x3 pair'}
        else:
            pair, path = saved_fast_pair(name, trait, run)
            provenance[trait] = {'run': name, 'weights': path.name, 'kind': 'fixed LDpred2-auto pair'}
        models.extend(pair)
        entries.extend([(trait, arm) for arm in ('snv_indel', 'with_sv')])
        chromosomes, starts, digests = extra.store_layout(run)
        layouts.append((chromosomes, starts.tolist(), digests))
    if any(layout != layouts[0] for layout in layouts):
        raise ValueError('Replication models index different site layouts')
    (out / 'provenance.json').write_text(json.dumps(provenance, indent=2) + '\n')
    # Independent score-path validation on discovery people, with no new outcome used.
    extra.check(argparse.Namespace(run='aou-x3-o1', batch='batch-001', workers=args.workers, chromosomes=None))
    table = extra.weight_table(models)
    run = runs['chronic_kidney_disease']
    with DosageStore.open(_final_store(run)) as store:
        structural = structural_records(store.variant_table)[table.rows]
    pattern = ('gs://long-reads-phase-2-imputation-staging-wb-rapid-tomato-9816/'
               '50k-tsps-finalized-vcfs/finalized-vcfs/teaspoons_50001_batch_{batch}.{chromosome}.vcf.gz')
    batches = [str(i) for i in range(5)]
    samples, scores, found, metadata = extra.score_callset(pattern, batches, run, table, args.workers, collect_info=True)
    extra.report_found('fixed score records', table, found, [trait + '/' + arm for trait, arm in entries])
    original = original_drift(run, table)
    widths = np.array([len(samples[batch]) for batch in batches])
    later = np.average(np.stack([metadata[batch] for batch in batches]), axis=0, weights=widths)
    sequencing = [name for batch in batches for name in samples[batch]]
    research = np.array([mapping.get(name, '') for name in sequencing])
    mapped = research != ''
    if len(set(research[mapped])) != mapped.sum():
        raise ValueError('Mapped callset contains duplicate research IDs')
    all_scores = np.concatenate([scores[batch] for batch in batches], axis=1)
    place = {person: index for index, person in enumerate(research) if person}
    for trait, name in specifications:
        columns = [entries.index((trait, arm)) for arm in ('snv_indel', 'with_sv')]
        member = table.member[:, columns].any(axis=1)
        complete = np.all(np.isfinite(original), axis=0) & np.all(np.isfinite(later), axis=0)
        if np.any(member & ~complete):
            raise ValueError('Incomplete required batch-drift metadata; inference stopped')
        drift = batch_drift(original[0, member], later[0, member], original[1, member], later[1, member], structural[member])
        print(trait + ': pre-specified batch drift', flush=True)
        print('\n'.join(drift_lines(drift)), flush=True)
        own = runs[trait]
        cohort = _read_cohort(own)
        rows = extra.new_people_rows(own, cohort.trait_names.index(trait), list(research[mapped]), excluded)
        print(trait + ': ' + ', '.join(key + ' ' + shown(value) for key, value in rows['counts'].items()), flush=True)
        selected = np.array([place[person] for person in rows['ids']], dtype=np.int64)
        arms = all_scores[np.ix_(columns, selected)]
        n, cases = len(selected), int(np.sum(rows['target'] == 1))
        if n < 40 or min(cases, n - cases) < 20:
            print(trait + ': insufficient people or outcome groups; inference suppressed', flush=True)
            continue
        np.savez(out / (trait + '.private.npz'), ids=rows['ids'].astype(str), target=rows['target'],
                 covariates=rows['covariates'], families=rows['families'], baseline=arms[0], candidate=arms[1])
        result = replicate(rows['target'], rows['covariates'], rows['families'], arms[0], arms[1], seed=1)
        lines = report(result, trait)
        (out / (trait + '.aggregate.txt')).write_text('\n'.join(drift_lines(drift) + lines) + '\n')
        print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        # Never export exception values or a traceback from participant processing.
        print('Replication stopped: ' + type(error).__name__, flush=True)
        raise SystemExit(1)
