"""Prepare common chr22 SNP fingerprints entirely inside the AoU app workspace.

No raw subprocess output, genotypes, sample names, or per-variant data is printed.
The source paths are release-specific; all intermediate files stay in --work.
Use the existing discovery BCF solely to select well-imputed common SNP sites,
then extract research-ID-keyed CDR ACAF genotypes and new-callset genotypes.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from pathlib import Path
import subprocess
import time

import numpy as np

from benchmarks.bench_real.genotype_identity import shown


def command(args, **kwargs):
    result = subprocess.run(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kwargs)
    if result.returncode:
        error = result.stderr.decode(errors='replace') if isinstance(result.stderr, bytes) else result.stderr
        known = {label: needle.lower() in error.lower() for label, needle in
                 [('missing_path', 'No such file'), ('access_denied', 'Permission denied'),
                  ('unknown_tag', 'not defined in the VCF header'), ('authentication', '401'),
                  ('authorization', '403'), ('memory', 'out of memory')]}
        print(json.dumps({'command_failed': Path(args[0]).name, 'categories': known}), flush=True)
        raise RuntimeError("Workspace extraction command failed")
    return result.stdout


def prepare(work):
    import pgenlib
    work.mkdir(parents=True, exist_ok=True)
    os.umask(0o077)
    home = Path.home()
    cdr = home / 'workspace/vwb-aou-datasets-controlled/v9'
    reference = cdr / 'wgs/short_read/snpindel/acaf_threshold/pgen/acaf_threshold.chr22'
    old = home / 'mnt/imp/deliverables/batch-001/aou2_50k_init.batch-001.chr22.prod-05fea60.glimpse2.popped.bcf'
    pattern = ('gs://long-reads-phase-2-imputation-staging-wb-rapid-tomato-9816/'
               '50k-tsps-finalized-vcfs/finalized-vcfs/teaspoons_50001_batch_{batch}.chr22.vcf.gz')
    state = work / 'prepare_status.json'
    def status(stage, **values):
        state.write_text(json.dumps({'stage': stage, **values}) + '\n')
        print(json.dumps({'stage': stage, **values}), flush=True)
    os.environ['GCS_OAUTH_TOKEN'] = command(['gcloud', 'auth', 'print-access-token'], text=True).strip()
    os.environ['GCS_REQUESTER_PAYS_PROJECT'] = os.environ['GOOGLE_PROJECT']
    status('selecting_common_autosomal_sites')
    site_file = work / 'candidate_sites.private.json'
    if site_file.exists():
        sites = json.loads(site_file.read_text())
    else:
        source = str(old) if old.is_file() else pattern.format(batch=0)
        header = command(['bcftools', 'view', '-h', source], text=True)
        fields = set(re.findall(r'##INFO=<ID=([^,]+)', header))
        frequency_field = 'RAF' if 'RAF' in fields else 'AF'
        expression = (f'TYPE="snp" && N_ALT=1 && INFO/INFO>0.97 && '
                      f'INFO/{frequency_field}>0.2 && INFO/{frequency_field}<0.8')
        output = command(['bcftools', 'query', '-i', expression,
                          '-f', '%POS\t%REF\t%ALT\n', source], text=True)
        sites, last = [], -5000
        for line in output.splitlines():
            pos, ref, alt = line.split('\t')
            if int(pos) - last >= 5000 and len(ref) == len(alt) == 1 and ref in 'ACGT' and alt in 'ACGT':
                sites.append([int(pos), ref, alt])
                last = int(pos)
        site_file.write_text(json.dumps(sites))
    if len(sites) < 2048:
        raise ValueError('Insufficient common SNPs')
    bed = work / 'sites.private.bed'
    bed.write_text(''.join(f'chr22\t{p-1}\t{p}\n' for p, r, a in sites))
    compact = work / 'reference_subset'
    status('extracting_CDR_reference', candidate_sites=shown(len(sites)))
    if not compact.with_suffix('.pgen').exists():
        # PLINK output is intentionally captured; it can contain small counts.
        command(['plink2', '--pfile', str(reference), '--extract', 'bed0', str(bed),
                 '--snps-only', 'just-acgt', '--max-alleles', '2', '--make-pgen',
                 '--out', str(compact), '--threads', '8', '--memory', '10000'])
    wanted = {(p, r, a) for p, r, a in sites}
    selected = []
    with compact.with_suffix('.pvar').open() as handle:
        rows = (line for line in handle if not line.startswith('##'))
        for index, row in enumerate(csv.DictReader(rows, delimiter='\t')):
            key = (int(row['POS']), row['REF'], row['ALT'])
            if key in wanted:
                selected.append((index, key))
    with compact.with_suffix('.psam').open() as handle:
        rows = list(csv.DictReader(handle, delimiter='\t'))
    ids = np.array([row.get('#IID', row.get('IID', '')) for row in rows])
    if any(not x for x in ids) or len(set(ids)) != len(ids):
        raise ValueError('Reference sample IDs are missing or repeated')
    genotypes = np.empty((len(ids), len(selected)), dtype=np.uint8)
    with pgenlib.PgenReader(str(compact.with_suffix('.pgen')).encode()) as reader:
        if reader.get_raw_sample_ct() != len(ids):
            raise ValueError('Reference sample dimension mismatch')
        row = np.empty(len(ids), dtype=np.int8)
        for column, (index, key) in enumerate(selected):
            reader.read(index, row)
            genotypes[:, column] = np.where(row < 0, 3, row)
    good = np.zeros(genotypes.shape[1], dtype=bool)
    for start in range(0, genotypes.shape[1], 256):
        part = genotypes[:, start:start + 256]
        called = part != 3
        frequency = np.where(called, part, 0).sum(axis=0) / (2 * called.sum(axis=0))
        good[start:start + 256] = (called.mean(axis=0) >= .99) & (frequency >= .1) & (frequency <= .9)
    del called, part
    sites = [key for (_, key), keep in zip(selected, good) if keep]
    genotypes = genotypes[:, good]
    if len(sites) < 2048:
        raise ValueError('Insufficient common called reference sites')
    region = work / 'query_regions.private.tsv'
    region.write_text(''.join(f'chr22\t{p}\n' for p, r, a in sites))
    key_index = {tuple(site): index for index, site in enumerate(sites)}
    query_arrays, query_ids, quality = [], [], []
    encoding = {'0/0': 0, '0|0': 0, '0/1': 1, '1/0': 1, '0|1': 1, '1|0': 1, '1/1': 2, '1|1': 2}
    for batch in range(5):
        status('extracting_new_callset', completed='in progress')
        os.environ['GCS_OAUTH_TOKEN'] = command(['gcloud', 'auth', 'print-access-token'], text=True).strip()
        path = pattern.format(batch=batch)
        names = command(['bcftools', 'query', '-l', path], text=True).split()
        data = np.full((len(names), len(sites)), 3, dtype=np.uint8)
        q = np.zeros(len(sites))
        error_file = work / 'bcftools.private.stderr'
        with error_file.open('w') as error:
            process = subprocess.Popen(['bcftools', 'query', '-R', str(region),
                                        '-f', '%POS\t%REF\t%ALT\t%INFO/INFO[\t%GT]\n', path],
                                       stdout=subprocess.PIPE, stderr=error, text=True)
            seen = set()
            for line in process.stdout:
                fields = line.rstrip('\n').split('\t')
                key = (int(fields[0]), fields[1], fields[2])
                column = key_index.get(key)
                if column is None:
                    continue
                if column in seen:
                    raise ValueError('Repeated matching VCF site')
                seen.add(column)
                if len(fields) - 4 != len(names):
                    raise ValueError('VCF sample dimension mismatch')
                data[:, column] = np.fromiter((encoding.get(x, 3) for x in fields[4:]), dtype=np.uint8, count=len(names))
                q[column] = float(fields[3]) if fields[3] != '.' else 0
            if process.wait():
                raise RuntimeError('New VCF extraction failed')
        query_arrays.append(data)
        query_ids.extend(names)
        quality.append((q >= .97) & ((data != 3).mean(axis=0) >= .99))
    usable = np.logical_and.reduce(quality)
    if usable.sum() < 2048:
        raise ValueError('Insufficient common high-quality sites across batches')
    # Deterministic thinning spans the chromosome without looking at matches.
    columns = np.flatnonzero(usable)
    columns = columns[np.linspace(0, len(columns)-1, min(4096, len(columns)), dtype=int)]
    variants = np.array([f'chr22:{sites[i][0]}:{sites[i][1]}:{sites[i][2]}' for i in columns])
    np.savez(work / 'query.private.npz', ids=np.array(query_ids), variants=variants,
             genotypes=np.concatenate([part[:, columns] for part in query_arrays]))
    np.savez(work / 'reference.private.npz', ids=ids, variants=variants, genotypes=genotypes[:, columns])
    status('fingerprints_ready', query_people=shown(len(query_ids)), reference_people=shown(len(ids)), sites=shown(len(columns)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work', type=Path, required=True)
    args = parser.parse_args()
    if not Path('/home/jupyter/workspace/vwb-aou-datasets-controlled').is_dir():
        raise RuntimeError('Preparation restricted to AoU workspace')
    prepare(args.work)


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print('Fingerprint preparation failed: ' + type(error).__name__, flush=True)
        raise SystemExit(1)
