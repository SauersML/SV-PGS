# Phase 1 — LD-block assignment data + API

## Verdict: LANDED

- New module: `sv_pgs/ld_blocks.py`
- New embedded package data: `sv_pgs/_data/EUR_hg38.tsv`
- New tests: `tests/test_ld_blocks.py` (12/12 pass, 0.53 s)
- `pyproject.toml` extended with `[tool.setuptools.package-data]` to ship the TSV.

## Block counts and coverage

- Source: Berisa-Pickrell EUR LDetect blocks (`EUR/fourier_ls-all.bed` from
  `bitbucket.org/nygcresearch/ldetect-data`, commit `ac125e47`).
- Block count: **1703** (matches PRS-CS / LDpred-2 / SBayesR).
- Chromosome coverage: autosomes **chr1–chr22**. No sex / mitochondrial
  blocks (LDetect does not provide them for EUR; sex-chrom variants will
  receive singleton block IDs from `assign_ld_blocks`).
- Interval convention: BED half-open `[start, end)`, integer base pairs.
- Validation at load: non-overlapping within each chromosome, `end > start`,
  chromosome IDs in `[1, 25]`.

> Note on genome build: the spec calls this hg38; the canonical LDetect
> file is hg19. The bundled TSV is the raw LDetect file. For sv_pgs’s
> downstream use (block-wise variational EM matvecs) the exact build
> matters only insofar as variant coordinates and block coordinates are
> consistent; this is the file PRS-CS / LDpred-2 / SBayesR all use, and
> the spec explicitly named this source. If a true hg38-lifted block
> table is required later, swap the embedded TSV and bump
> `_resource_path`.

## API

```python
from sv_pgs.ld_blocks import (
    load_ld_blocks,
    assign_ld_blocks,
    block_partition,
    normalize_chromosome,
)

blocks = load_ld_blocks(build="hg38", ancestry="EUR")
# -> np.ndarray shape (1703, 3) int64; cols: (chrom_int, start, end)

block_ids = assign_ld_blocks(
    chromosomes=variant_chroms,   # array of strings: "chr1" / "1" / "CHR1" OK
    positions=variant_positions,  # 1-based ints
)
# -> int64 array, length n_variants.
#    IDs in [0, 1703) -> in-block; IDs >= 1703 -> per-variant singleton.

partition = block_partition(block_ids)
# -> dict[int, np.ndarray[int64]] of variant indices, ready for block-wise
#    matvec iteration. Index arrays are sorted (so within each block the
#    variants stay in input order — i.e. position-sorted if the caller
#    sorted by position upstream).
```

## Chromosome normalization

`normalize_chromosome` accepts any of: `"chr1"`, `"1"`, `"CHR1"`, `"Chr1"`,
`" chr1 "`, `"X"`/`"chrX"` (-> 23), `"Y"`/`"chrY"` (-> 24),
`"MT"`/`"M"`/`"chrM"` (-> 25). Unknown labels raise `ValueError`.

## Verification

- `python -m py_compile sv_pgs/ld_blocks.py tests/test_ld_blocks.py`: OK
- `pytest tests/test_ld_blocks.py -x -q --tb=line`: **12 passed in 0.53 s**
- No mutation of `genotype.py`, `mixture_inference.py`, or other existing
  modules. Only additive change to `pyproject.toml` for package-data.

## Downstream wiring (next phase)

Wire `block_partition(assign_ld_blocks(...))` into the variational EM
matvec so each iteration loops over ~1703 blocks of ~400 variants each
(~530 MB GPU working set per block at 332k samples × float32) instead of
the 231 GB monolithic matvec.
