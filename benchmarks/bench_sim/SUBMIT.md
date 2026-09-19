# Submitting a method to bench-sim

bench-sim runs your method. You don't run the sealed benchmark and never see its seeds, truths or test phenotypes.

## Contract
Put one Python file at `/projects/standard/hsiehph/sauer354/svpgs-team/bench-sim/submissions/<lane>/<name>.py`, then SendMessage `bench-sim` with its path and the venv it needs. The venv defaults to `venv-cpu`; `venv-gpu` is available, and a lane-private venv needs its path.

```python
def fit(train):            # -> model
    ...
class Model:
    def score(self, test):  # -> np.ndarray [test.n_samples]  or  {"total": ..., "structural": ...}
        ...
```

- **`train`:**
  - `train.codes(rows)`: uint8 observed codes, shape [len(rows), n_train_samples]; dosage = code / 127, every record's GLIMPSE2 DS;
  - `train.variants`: a dict of public per-record arrays: pos, cm, cls (0 SNV, 1 INDEL, 2 TR, 3 SV), len_change (signed), ref_len, alt_len, in_gene, in_exon, log_tss_distance, in_repeat, log_sv_length, and imputation_info (the mean GLIMPSE2 INFO);
  - `train.covariates` [n_train, 13], with names in `train.covariate_names`: sex, age (standardized), batch, pc1–pc10;
  - `train.phenotype`, `train.trait_type` ("quantitative" or "binary"), `train.prevalence` (binary only), and `train.cores`.
- **`test`:** the same `codes(rows)`, `variants` and `covariates`, for test samples, with no phenotype.
- **`structural`:** optional. It's the part of your prediction carried by TR+SV records, used for SV credit.
- **Records:** chr22 only for v1, 590,623 records at donor MAC ≥ 3 (487,545 SNV, 70,482 INDEL, 30,585 TR, 2,011 SV), with 50,000 samples (40,000 train and 10,000 test).

## Rules
- **Matched compute:** one runq task per scenario, with 16 cores and at most 1 GPU (declare it), and a 4-hour wall limit per scenario. Wall time and peak RSS are recorded.
- **No hand-tuning on the benchmark.** Dev scenarios (`bench-sim/dev/`, public, 24 of them, with full truth) are for development. The 64 sealed test scenarios are run once per method version.
- **Report back:** every scenario's paired difference against the reference arms (`ridge_inf_simple`, `ridge_inf_all`, `oracle_observed`), losses included, plus calibration slope, R² per ancestry group, SV-credit error and the per-parameter breakdowns.
