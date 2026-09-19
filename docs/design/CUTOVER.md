# Cutover: deleting the old path (C2–C8)

This inventory was measured on main at `c2a9443` with a static import graph: Python's `ast` over `sv_pgs/`, `tests/` and `scripts/`, text analysis only, with nothing imported or run.
- Symbols are named, not line-numbered, because line numbers drift.
- C0 (dead code, `21cdec3`) and C1 (moves into the new modules, e.g. `f6af37f`) are done.

## Progress
The pre-cutover tree is tagged `archive/2026-09-19/old-path-final` (`f025cce`).

| Step | State |
|---|---|
| B1, B2 | landed `6a85d17`: the package root exports only `ModelConfig`, `TraitType`, `VariantClass`, `VariantRecord`; kept modules load no JAX and leave `CUPY_TF32` unset (`test_package_import`) |
| B4 | landed `664dea7`: `sv_pgs/cohort.py` (ancestry PCs, indicators, KING-component folds stratified by half × ancestry, one full-rank C, NaN-masked multi-trait targets) |
| B3 | ruled: until the engine's Stage 2 driver lands, the CLI keeps only commands backed by kept modules; old fit commands are removed, not stubbed |
| B5, B6 | ruled: the engine owns the certificate artifact and logistic EP; the old binary code goes with C5 |
| B7 | ruled: owned by the deslop-fit lane |
| C2 | landed: the CLI keeps the phenotype commands and `version`; `run`, `run-all-of-us`, `evaluate-all-of-us`, `doctor` and `run.sh` are gone (with `test_io`, `test_cli_doctor`) |
| C3 | landed: `aou_runner`, `aou_storage`, `evaluate`, `pipeline`, `benchmark`, `preflight`, `__main__` and `scripts/analyze_model.py`; `progress.py` is `log`/`elapsed`/`mem`; CI keeps only the full-suite shards (no subset job, no timeout or retention constants) |
| C4 | landed: `model.py`, `runtime_policy.py`; `artifact.py` stays until the engine extends or replaces it; `test_prior_level_without_variance_refresh` went here (its helper was in `test_prediction_accuracy`) |
| C5 | pending |
| C6 | pending |
| C7 | pending |
| C8 | pending |

## Size
- **`sv_pgs/`:** 65,304 lines.
  - Old path: 41 modules, 48,616 lines (74%). `cli.py`, `__init__.py` and `__main__.py` survive in trimmed form.
  - Kept: 36 modules, 16,688 lines, plus the engine modules still to land. Five of them serve only the old path and go too unless something adopts them: `preflight`, `path_policy`, `sample_table`, `inference`, `numeric` (1,023 lines).
- **`tests/`:** 156 files.
  - Deleted or rewritten in C3–C7: 114 files, about 34k lines.
  - Kept: 42 files, most unchanged.
- **Dependency direction:**
  - No kept module imports an old module at top level.
  - Every remaining link is either one of the blockers below, or runs from old to new. The old modules import the C1 moves back from their new homes.

## Blockers: kept code that still reaches the old path
- **B1 — `sv_pgs/__init__.py` imports the old path eagerly** (`benchmark`, `io`, `model`, `pipeline`).
  - So `import sv_pgs.<anything>`, including every kept module and `tests/conftest.py`'s autouse fixture, also loads `model`, `mixture_inference`, `genotype` and `_jax`.
  - `_jax` then imports JAX, turns on x64, rewrites the `XLA_*` environment variables and sets `CUPY_TF32=1` by default for the whole process.
  - Kept GPU code is fp64 or exact int8, so no result changes today. But the kept path is not independent of the old one until `__init__` is trimmed (C2).
- **B2 — two kept tests import a re-export from `genotype`:**
  - `tests/test_fast_scoring.py` and `tests/test_code_products_cuda.py` take `_try_import_cupy` from `sv_pgs.genotype`; it should come from `sv_pgs.compute_budget`.
  - `test_fast_scoring.py` also imports `TraitType` from the package root.
  - Repoint all three in C2, or these tests die with `genotype` in C6.
- **B3 — no kept entry point.**
  - The only fit drivers are `model.BayesianPGS.fit` and `pipeline.run_training_pipeline`, reached from `cli run` and `aou_runner.run_all_of_us`.
  - Nothing yet takes the store → Stage 0 → Stage 1/2 → scoring path. That is the engine's Stage 2 driver (next section).
- **B4 — no kept cohort builder.** The model needs:
  - covariates C: sex, age terms, PCs, a pipeline-half indicator and a cohort indicator;
  - kinship-grouped folds over both store halves;
  - a multi-trait Y[n, T].
  The only implementations are in the old path, and each fails the design:
  - `aou_runner.merge_pcs_into_sample_table`;
  - `aou_runner._split_merged_sample_table`: a sha256 split with no kinship handling;
  - `aou_runner._expand_one_hot_covariates`: race/ethnicity one-hots;
  - `sample_table._build_sample_table`: one target, float32.
  - **No lane owns this yet.** Port the PC-merge logic before C3 deletes `aou_runner`, or recover it later from git history.
- **B5 — no kept artifact.**
  - `artifact.ModelArtifact` is written only by `BayesianPGS.export`.
  - MODEL.md §4 requires the certificate to be recorded in the artifact, so the engine either extends `artifact.py` or replaces it.
- **B6 — binary traits have no owner on the new path.**
  - MODEL.md §4 specifies logistic EP with a converged Gauss–Hermite predictive. `fast_scoring` already implements the predictive.
  - The engine brief covers only the variant-side EP. The old binary machinery must not be deleted before logistic likelihood sites exist on the new path:
    - `tr_newton.py`;
    - the Pólya–Gamma, Laplace and `_fit_binary_alpha_with_offset` code in `mixture_inference`.
  - This gates only the binary half of C5.
- **B7 — prior code that violates the no-hand-chosen-prior rule and has no replacement yet.**
  - `prior_design._scale_model_penalty` gives the annotation coefficients a fixed ridge (`scale_model_ridge_penalty` = 1.0, `type_offset_penalty` = 2.0).
  - `_support_bounded_member_log_scale_predictions` clamps scales to a fixed range (`prior_scale_floor`/`_ceiling` = 1e-6/10).
  - The engine's EB step replaces both (next section).

## The engine and the cutover
The engine lane is building a production variant-side inference module in `sv_pgs/`, independent of `tests/ep_eb_reference.py`. Planned names, from `team/e2e/STATUS.md`: `scale_mixture_prior.py`, and a Stage 2 driver `full_data_fit.py`. It lands before any deletion.

**Rules so the two can be sequenced:**
- The engine imports no old module.
- Its knobs are new `ModelConfig` keys, never reuses of old keys, so C8 can delete every old key.
- Its result type and its artifact carry the certificate.

**Old-path pieces the engine replaces (marked E below).** Delete each only after its replacement is on main.

| Old piece | Replaced by |
|---|---|
| `mixture_inference.fit_variational_em` (CAVI/EM loop), `_fit_collapsed_posterior`, the sample-space / Nyström / CG-Lanczos / workset routes | EP site updates, plus the Stage 2 driver on `exact_polish.FullDataGaussian` |
| `mixture_inference._update_local_scales`, `_gig_moment`, `_update_tpb_shape_vectors`, `_initialize_scale_model`, `_update_scale_model`, `_calibrate_initial_global_scale`, the Anderson/θ packing | Tilted moments of the scale mixture; the EB update of the mixing density, annotation coefficients and smoothing weights |
| `prior_design._scale_model_penalty`, `_metadata_baseline_scales_from_coefficients`, `_metadata_baseline_scales_from_log_predictions`, `_support_bounded_member_log_scale_predictions`, `_effective_prior_variances`, `_scale_state_reduced_prior_variances` (B7) | The same EB update (learned smoothing weights, data-driven scale range) |
| `elbo.py`, and the noise updates inside `fit_variational_em` | The noise-variance EB |
| `forcing_sequence.py`, `precision_policy.py`, the convergence diagnostics in `mixture_inference`, `ModelConfig.allow_nonconverged_export` | The certificate: Newton decrement plus ‖XΔμ‖/‖Xμ‖ |
| `inference.VariationalFitResult` | The engine's result type |
| `artifact.ModelArtifact` (B5) | Extended with the certificate, or replaced |
| `genotype` (`RawGenotypeMatrix` and its backends), `bitpacked_matrix`, `plink`, `io` loaders | Genotype blocks read from the store: `dosage_store` plus `code_products.CodeBlockTile` |
| `model.BayesianPGS.fit`, `pipeline.run_training_pipeline`, `cli run` | The Stage 2 driver started from the prior, and the end-to-end synthetic test |
| `tr_newton.py`, the binary code in `mixture_inference` (B6) | Logistic EP sites; owner not yet assigned |

**Kept modules with no kept importer yet.** The engine is expected to import them; any still unused when their step comes are dead code and go in that step:
- `anderson.py`: the EFS + Anderson hyper step. C5 deletes it only if it is still unused.
- `prior_design.py`: the d_j design builders `_build_prior_design`, the feature specs, spline, factor and nested encoders, and `collapse_tie_groups`. The E functions above go.
- `hyperprior_pooling.py`: cross-trait pooling of the level and θ.
- `numeric.py` (`stable_sigmoid`, for logistic sites): C5 deletes it if still unused.
- `artifact.py`: C4 deletes it if it is replaced.

**Old tests whose intent the engine's tests should carry:**
- `test_one_prior_all_routes`: the final posterior is the Gaussian posterior under the reported prior;
- `test_optimizer_invariants`, `test_output_dtype`, `test_numerical_guards`;
- `test_convergence_export_contract_pinning`: a fit without a certificate is not exported;
- `test_prediction_accuracy`, `test_adversarial_high_dimensional`, `test_adversarial_limits`: accuracy on synthetic data;
- `test_e2e_synthetic`, `test_e2e_equivalence`, `test_e2e_scale`: replaced by the store-based end-to-end test.

## Order
- **After C2, nothing kept reaches the old path, so deletions go top-down.**
  - Each commit deletes whole files whose importers are already gone.
  - Surviving files are edited only where they call into the deleted set.
  - This moves the history inventory's themes: the marginal screen (old C3) goes with the genotype backends, because it lives in `io`, `preprocessing`, `model` and `bitpacked`. The old scorer (old C5) goes with `model.py`.
  - Checked mechanically: after each step, no remaining module or test imports a deleted one.
- **Pyproject and uv.lock edits in any step go through the env lane** (RESUME_BRIEF).

### C2 — flip the entry point (no module deletions)
- **Prerequisites:**
  - the engine and its Stage 2 driver are on main;
  - the e2e lane's store-based end-to-end synthetic test passes on main (full suite and CI);
  - B5 is resolved.
  - B4 is needed before the AoU pilot, not before C2.
- **Edit:**
  - `sv_pgs/__init__.py`: drop the imports and exports of `benchmark`, `io`, `model` and `pipeline`. Export `ModelConfig`, `TraitType`, `VariantClass`, `VariantRecord` and the engine's fit entry. Fix the docstring (B1).
  - `sv_pgs/cli.py`:
    - `run` now drives store → Stage 0 → Stage 2 → scoring through the engine;
    - drop the top-level imports of `io` and `pipeline`;
    - `run`'s old flags go: `--genotype-format`, `--variant-metadata`, `--marginal-screen-min-abs-z`, `--max-outer-iterations`, `--allow-nonconverged-export`;
    - its banner calls go: `jax_runtime_snapshot`, `gpu_memory_snapshot`, `log_autotune_banner`;
    - `doctor` becomes a `ComputeBudget` report, or is removed, with its `bitpacked.launch`, `bitpacked.smoke` and `preflight._probe_cupy_nvrtc` imports.
  - Tests:
    - `test_fast_scoring.py` and `test_code_products_cuda.py`: import from `sv_pgs.compute_budget` and `sv_pgs.config` (B2).
    - `test_package_import.py`: the new export list.
    - `test_io.py`: drop its five old-`run` CLI cases, or the whole file, since its modules go in C3–C6.
    - `test_cli_doctor.py`: rewrite with `doctor`, or delete.
- **Verify:**
  - `git grep -n 'from sv_pgs.genotype import _try_import_cupy' -- tests` prints nothing.
  - On MSI: `<venv>/bin/python -c "import sys, sv_pgs.dosage_store, sv_pgs.genotype_statistics, sv_pgs.exact_polish, sv_pgs.fast_scoring; bad = [m for m in sys.modules if m == 'jax' or m in ('sv_pgs.model', 'sv_pgs.genotype', 'sv_pgs._jax')]; assert not bad, bad"`.
  - Then the full suite and CI (see the end of this document).

### C3 — old orchestration and AoU runner
- **Prerequisite:** C2. Port B4's PC-merge logic first if the cohort builder will reuse it.
- **Delete:**
  - `aou_runner.py` (2,445), `aou_storage.py` (707), `evaluate.py` (576: the quasi-holdout over run-all-of-us outputs, superseded by `held_out_comparison.py`), `pipeline.py` (644), `benchmark.py` (155);
  - `run.sh` (853; it also runs pip, against SPEC) and `scripts/analyze_model.py` (1,091);
  - `preflight.py` (442), unless the Batch runbook adopts `check_aou_preflight`. After C2 its only caller is gone.
- **Edit:**
  - `cli.py`:
    - drop `run-all-of-us` (with `--disease`, `--trait`, `--all-diseases`, `--chromosomes`, `--variants`, `--variant-metadata`, `--n-pcs`, `--max-parallel-gpus`, `--dry-run`, `_run_dry_run`) and `evaluate-all-of-us`;
    - drop the imports of `aou_runner` (`_normalize_variants_choice`, `run_all_of_us`, `run_all_of_us_all_diseases`) and `evaluate`;
    - the AoU entry is now the Batch runbook (COMPUTE.md);
    - keep `list-all-of-us-diseases`, `prepare-all-of-us-disease`, `list-all-of-us-traits`, `prepare-all-of-us-trait`, `census-all-of-us-traits` and `version`.
  - `progress.py`: delete the functions whose last callers are gone:
    - `log_autotune_banner`, which reaches `genotype` and `_jax`;
    - `device_summary`, which reaches `bitpacked.launch` and `gpu_scheduler`;
    - `jax_runtime_snapshot` and `gpu_memory_snapshot` (JAX);
    - `set_log_file` and `start_heartbeat` too, unless the runbook adopts them.
    - After this, `progress.py` imports nothing from the old path.
  - `config.py`: delete `BenchmarkConfig`.
  - `__main__.py`: docstring. `tests/test_cli.py`: use a surviving command.
  - CI `cpu-tests` job: drop `tests/test_aou_storage.py` (and `tests/test_preflight.py` if `preflight` goes).
  - pyproject (env lane): drop `matplotlib`.
- **Tests:**
  - Delete: `test_aou_storage`, `test_atomic_gcs_download_pinning`, `test_benchmark`, `test_benchmark_constant_target_boundaries_pinning`, `test_benchmark_single_class_pinning`, `test_e2e_equivalence`, `test_e2e_scale`, `test_e2e_synthetic`, `test_evaluate_input_paths`, `test_evaluate_survey_self_report`, `test_io`, `test_multi_gpu_scheduler_pinning`, `test_quasi_holdout_score_selection_pinning`, `test_regression_runtime_safety`, `test_select_score_column_empty_boundaries_pinning`; also `test_preflight`, if `preflight` goes.
  - Rewrite:
    - `test_all_of_us` and `test_all_of_us_measurements`: keep the `all_of_us` phenotype and SQL cases in one `test_all_of_us_phenotypes.py`; drop the `aou_runner`, `cli run-all-of-us`, `io` and `mixture_inference` cases.
    - `test_aou_dummy_trap_fix`: carry its intent (no dummy-variable trap) into the cohort builder's tests.
    - `test_prediction_cache_id_safety_bugfixes`: re-implement its cohort-mismatch guard in `test_fast_scoring`.
    - `test_convergence_export_contract_pinning`: move to the engine (E).
    - `test_no_caps`: keep its source scans ("no caps on variants or samples") over the kept module list.
- **Verify:**
  - `git grep -n -P 'sv_pgs\.(aou_runner|aou_storage|evaluate|pipeline|benchmark)\b|from sv_pgs import .*\b(aou_runner|aou_storage|evaluate|pipeline|benchmark)\b' -- sv_pgs tests scripts .github` prints nothing.
  - Then the full suite, vulture and CI.

### C4 — `model.py`: the old fit orchestration and the old scorer
- **Prerequisites:** C3 and B5.
- **Delete:**
  - `model.py` (3,872): `BayesianPGS`, `FittedState`, the scorer methods (`decision_function`, `predict_proba`, …), the fit caches, the checkpoint persistence and the marginal-z helpers;
  - `runtime_policy.py` (254; imported only by `model`);
  - `artifact.py` (361), if the engine replaced it (E). Otherwise keep it and its four tests.
- **Edit:** none. Every importer of `model` is gone after C2/C3.
- **Tests:**
  - Delete: `test_adversarial_high_dimensional`, `test_adversarial_limits`, `test_autotune_oom_safety`, `test_cache_corruption_safety` (keep its artifact case if `artifact.py` stays), `test_marginal_z_concat_fast_path`, `test_model`, `test_prediction_accuracy`, `test_scoring_bitpacked_fast_path`.
  - The scorer cases the history inventory marked for moving are already in `test_fast_scoring`: `predictive_intercept_shift` and `posterior_predictive_probability`.
- **Verify:** `git grep -n -P 'sv_pgs\.(model|runtime_policy)\b|from sv_pgs import .*\b(model|runtime_policy)\b' -- sv_pgs tests` prints nothing; then the full suite, vulture and CI.

### C5 — the old inference core (GIG/δ mean field, SVI, PG-IRLS / TR-Newton, solvers)
- **Prerequisites:**
  - C4;
  - every E row above is on main;
  - B6, for the binary files. If B6 is open, split: C5a takes everything except `tr_newton.py` and the binary helpers, and C5b takes those after logistic EP lands. `mixture_inference.py` then shrinks to its binary part in C5a.
- **Delete:**
  - `mixture_inference.py` (13,147), `linear_solvers.py` (1,451), `tr_newton.py` (800), `elbo.py` (175), `forcing_sequence.py` (136), `precision_policy.py` (93), `inference.py` (35), `gpu_scheduler.py` (216);
  - `numeric.py` (46) and `anderson.py` (127), only if the engine does not import them.
- **Edit:**
  - `prior_design.py`: delete the E functions (B7);
  - `config.py`: none yet (C8).
- **Tests:**
  - Delete: `test_anderson_actually_accelerates`, `test_binary_laplace_variance`, `test_binary_tr_newton_path`, `test_block_shuffle`, `test_cg_workset_resident_cache_pinning`, `test_checkpoint_compat`, `test_elbo`, `test_elbo_zero_variance_boundaries_pinning`, `test_end_to_end_smoke`, `test_exact_inverse_diagonal_cost`, `test_forcing_sequence`, `test_forcing_sequence_zero_grad_boundaries_pinning`, `test_gig_mean_small_argument`, `test_gpu_cg_no_stream_capture`, `test_gpu_cholesky_solve_no_factor_copy`, `test_gpu_cholesky_solve_zero_covariates`, `test_gpu_memory_hygiene`, `test_gpu_scheduler`, `test_inference` (5,657), `test_lanczos_one_by_one_boundaries_pinning`, `test_ld_block_smoke`, `test_linear_solvers`, `test_linear_solvers_bugfixes`, `test_local_scales_collapsed`, `test_numerical_guards`, `test_one_prior_all_routes`, `test_optimizer_invariants`, `test_output_dtype`, `test_prior_level_without_variance_refresh`, `test_resume_continuity`, `test_resume_with_new_features`, `test_sample_space_operator_bitpacked`, `test_sample_space_operator_ld_sharded`, `test_sigma_e2_elbo`, `test_solve_spd_zero_rhs_boundaries_pinning`, `test_solver_controls_des`, `test_tpb_lbfgs`, `test_tr_newton`, `test_tr_newton_nonconvergence_bugfixes`, `test_tr_newton_zero_iter_boundaries_pinning`, `test_variance_state_no_stale_ema`, `test_warm_start_api`, `test_warm_start_category_reconciliation`.
  - `test_stable_sigmoid_log1p_boundaries_pinning`: keep its `numeric` cases if `numeric.py` stays.
  - The intent carried by the engine's tests is listed above.
- **Verify:** `git grep -n -P 'sv_pgs\.(mixture_inference|linear_solvers|tr_newton|elbo|forcing_sequence|precision_policy|inference|gpu_scheduler)\b|from sv_pgs import .*\b(mixture_inference|inference|elbo|tr_newton)\b' -- sv_pgs tests` prints nothing; then the full suite, vulture and CI.

### C6 — genotype backends and the marginal |z| screen
- **Prerequisite:** C5 (a).
- **Delete:**
  - `genotype.py` (7,097), `io.py` (4,202), `preprocessing.py` (1,721; including the marginal-z functions, `build_tie_map` and `compute_variant_statistics`), `screening_pipeline.py` (971);
  - `plink.py` (1,059), `mmap_reader.py` (354), `gcsfuse_staging.py` (1,344), `gds.py` (108);
  - `bitpacked/` (10 files, 3,593), `bitpacked_loader.py` (1,349), `bitpacked_matrix.py` (641), `bitpacked_profile.py` (159);
  - `ld_blocks.py` (263), `ld_block_partition.py` (127) and `sv_pgs/_data/EUR_hg38.tsv`. EUR blocks don't fit this admixed cohort; Stage 0's `ld_partition` replaces them.
  - `sample_table.py` (182).
  - `path_policy.py` (318), unless the store reader or runbook adopts `assert_hot_local_path`.
- **Edit:**
  - `data.py`: delete `VariantStatistics` and `PreparedArrays`.
  - `tie_map.py`: delete `_empty_tie_map` (its callers are `model` and `preprocessing`).
  - `_typing.py`: delete `I8Array` if unused.
  - `compute_budget.py`: delete what vulture then flags (`_cupy_runtime_diagnostic`, `_nvidia_driver_diagnostic` and `_detect_available_host_ram_bytes` were only `genotype`'s).
  - `config.py`: delete `use_ld_blocks`, `ld_block_*`, `genotype_backend`, `use_mmap_bed`, `stage_gcsfuse_locally`, `marginal_screen_min_abs_z` and `marginal_screen_protect_sv`, with their validators. Keep `minimum_minor_allele_frequency`: Stage 0 reads it.
  - `tests/conftest.py`: drop the `bitpacked_profile` / `bitpacked_loader` autouse fixtures and `make_fake_cupy` with its last users.
  - Docstrings: `dosage_store.py` (pinned-pool rationale cites the bitpacked cache) and `genotype_buffers.py` (cites `genotype._try_import_cupy`).
  - CI `cpu-tests` job list: drop `test_bitpacked_lut`, `test_bitpacked_cpu_reference`, `test_bitpacked_launch`, `test_bitpacked_package_smoke`, `test_bitpacked_engagement`, `test_gds`, `test_mmap_reader`, and `test_path_policy` if that module goes.
  - pyproject: drop `sv_pgs = ["_data/*.tsv"]` package data.
- **Tests:**
  - Delete: all 14 `test_bitpacked_*`, `test_e2e_sv_joint`, `test_gds`, `test_genotype` (2,948), `test_genotype_bugfixes`, `test_gpu_materialization_budget_bugfixes`, `test_ld_blocks`, `test_marginal_screen_one_pass`, `test_marginal_z_covariate_pinning`, `test_marginal_z_degenerate_boundaries_pinning`, `test_mmap_reader`, the four `test_plink_*`, `test_preprocessing_bugfixes`, `test_region_parse_worker`, `test_screening_gpu_gather`, `test_screening_pipeline`, `test_v100_budget_pinning`, `test_variance_single_sample_boundaries_pinning`, `test_variant_typing_parity`, `test_vcf_record_filtering`.
  - Rewrite:
    - `test_preprocessing`: move its `collapse_tie_groups` cases next to `prior_design`.
    - `test_tie_map_all_missing_boundaries_pinning` and `test_hardcall_tie_sign_flip_pinning`: port the all-missing and sign-flip tie cases to `test_genotype_statistics`, if not already covered.
- **Verify:** `git grep -n -P 'sv_pgs\.(genotype|io|preprocessing|screening_pipeline|plink|mmap_reader|gcsfuse_staging|gds|bitpacked|bitpacked_loader|bitpacked_matrix|bitpacked_profile|ld_blocks|ld_block_partition|sample_table)\b' -- sv_pgs tests .github pyproject.toml` prints nothing; then the full suite, vulture and CI.

### C7 — JAX and old-only dependencies
- **Prerequisite:** C6. After it, `_jax` has no importers.
- **Delete:** `_jax.py` (225). It set `CUPY_TF32` and the device pin implicitly; if any kept code needs either, make it explicit in `ComputeBudget`.
- **Edit:**
  - `_typing.py`: delete `JaxArray` and its note.
  - pyproject (env lane):
    - drop `jax` and `jaxlib`, `jax[cuda12]` in the `gpu` extra, and the `jax`/`jaxlib` mypy overrides;
    - drop `psutil`: it is imported nowhere today, since C1a;
    - move `scikit-learn` to the dev group: only `tests/test_held_out_comparison.py` uses it, for `roc_auc_score`;
    - keep `cyvcf2` (`gatksv_source`, `store_converter`), `pandas` (`external_annotations`), `google-cloud-bigquery` (`all_of_us`), `google-crc32c`, `zstandard`, `threadpoolctl`, and in the `gpu` extra `cupy-cuda12x` plus `nvidia-cuda-nvrtc-cu12`.
  - uv.lock: regenerated by env.
  - CI: drop `JAX_PLATFORMS: cpu`.
- **Tests:** delete `test_jax_gpu_detection` and `test_jax_runtime`.
- **Verify:** `git grep -n -P '\bjax\b|jaxlib|_jax\b' -- sv_pgs tests pyproject.toml .github` prints nothing; then `uv lock --check` and the full suite on MSI in a venv rebuilt without JAX, then CI.

### C8 — config, CLI, docs
- **Prerequisite:** C7.
- **`config.py`:**
  - delete `DEFAULT_CLASS_LOG_BASELINE_SCALE`, `DEFAULT_CLASS_TPB_SHAPE_A` and `DEFAULT_CLASS_TPB_SHAPE_B`: hand-chosen asymmetric SV/SNV prior tables;
  - delete the methods `class_log_baseline_scales`, `class_tpb_shape_a`, `class_tpb_shape_b` and `structural_variant_classes`;
  - delete every `ModelConfig` key with no kept reader, with its validator in `__post_init__`:
    - `max_outer_iterations`, `convergence_tolerance`, `polya_gamma_minimum_weight`, `sigma_error_floor`, `global_scale_floor`, `global_scale_ceiling`;
    - `maximum_scale_model_iterations`, `tpb_hierarchical_prior_variance`, `maximum_tpb_shape_iterations`, `minimum_tpb_shape`, `maximum_tpb_shape`;
    - `max_inner_newton_iterations`, `binary_inner_tolerance`;
    - `linear_solver_tolerance`, `maximum_linear_solver_iterations`, `logdet_probe_count`, `logdet_lanczos_steps`, `exact_solver_matrix_limit`;
    - `posterior_variance_batch_size`, `posterior_variance_probe_count`, `beta_variance_update_interval`, `final_posterior_diagnostics`, `allow_nonconverged_export`;
    - `use_tr_newton_binary`, `cg_progress_interval`, `solver_wall_clock_budget_s`;
    - `sample_space_preconditioner_rank`, `validation_interval`, `validate_first_iteration`;
    - `stochastic_*`, `posterior_working_set_*`, `random_seed` (unless the engine's draws read it), plus the B7 keys once E lands: `prior_scale_floor`/`_ceiling`, `local_scale_floor`, `scale_model_ridge_penalty`, `type_offset_penalty`;
  - keep: `trait_type`, `minimum_scale`, `minimum_minor_allele_frequency`, `TraitType`, `VariantClass`, and the engine's own keys;
  - rewrite the docstring.
- **Tests:**
  - `test_tpb_prior.py` becomes `test_prior_design.py`: drop `TestClassSpecificTPBShapes`, `TestHierarchicalPooling`, the baseline and ridge cases of `TestMetadataScaleModel`, and `TestConfigValidation`; keep the design-matrix encoder cases.
- **Docs:**
  - `README.md`: rewrite around the new path. The All of Us quickstart, generic usage, GPU check, bitpacked smoke/bench, data and troubleshooting sections all describe the old path.
  - `docs/design/README.md` and HANDOFF.md: drop "the old path is still present".
  - pyproject `description` and the `__init__` docstring: no longer "multi-GPU joint empirical-Bayes GLM".
  - `.gitignore`: drop `.sv_pgs_cache/`, `diagnose_runs*.txt`, `falsify.txt`, `inspect_runs.txt`, `probe_*.txt` and `verify_fix.txt` once nothing writes them.
- **SPEC.md (lead's ruling):**
  - L4 "Very rare SVs will be filtered" becomes the information-based inclusion rule;
  - L14–15, the JAX rule, becomes an array-module rule.
- **Verify:** `<venv>/bin/python -m vulture sv_pgs tests --min-confidence 80` reports nothing new; then the full suite and CI.

## Verification commands (every step)
- **Text check on the laptop:** the step's `git grep` above prints nothing.
- **Full suite on MSI** (serialized, RESUME_BRIEF):
  `flock <team scratch>/fullsuite.lock taskset -c <slice> nice -n 10 timeout 3600 <venv>/bin/python -m pytest -q`, with 0 failed.
- **Dead code:** `<venv>/bin/python -m vulture sv_pgs --min-confidence 80`. Every finding is either deleted in the same step, or is an engine-pending kept module (listed above).
- **CI:** `gh run list -R SauersML/SV-PGS -L 5` shows green for the pushed sha.

## Stale in the history inventory (`team/history-miner/cutover/msg_to_build-lead_0015.md`, written against `4bc6dbc`)
- **P-a:** `exact_polish` importing the GIG/δ M-step. It now imports only `config`.
- **P-b:** `ld_space_fit` importing `_gig_moment`. The old Stage 1 is not on main; it's archived.
- **P-c:** exactness references using `fit_variational_em`.
  - `test_exact_polish` no longer does.
  - The dense EP-EB reference comes from the oracle lane; `test_fast_scoring` no longer compares against `BayesianPGS`.
- **P-d:** `dosage_store` importing `bitpacked_loader`'s pinned pool. The pool moved into `dosage_store` (C1b); the dependency now runs the other way.
- **P-e:** JAX `stable_sigmoid`. It is NumPy since `f6af37f`.
- **P-f:** the missing `build_prior_design_from_table`. `_build_prior_design` moved into `prior_design.py` (C1c) and stays the only design builder.
- **P-g, P-h:** open `fix/*` and `build/phenotypes` branches. All branches were deleted or archived on 2026-09-19, and the phenotypes landed (`61a8997`).
- **C1 remnants** (all old-only now; they go with their files in C5/C6):
  - `_calibrate_binary_intercept` is still in `mixture_inference`; `fast_scoring.predictive_intercept_shift` covers the kept path.
  - `build_tie_map` is still in `preprocessing`; Stage 0 builds tie maps with `tie_map_from_groups`.
  - The `_probe_cupy` duplicates are in `gpu_scheduler` and `preflight`.
- **C2 "fit → fast_fit.fit_fast":** `fast_fit.py` is archived and obsolete; it hard-coded prior constants. The target is the engine's Stage 2 driver.
- **KEEP items no longer kept:**
  - `model.py`'s `FittedState`, `export`/`load` and `_tie_group_export_weights`: `fast_scoring` no longer imports `model`, so it goes whole in C4.
  - The keys "read by `exact_polish` today": it reads only `trait_type`.
  - `aou_runner`'s covariate and split helpers: they fail the design (B4).
  - `evaluate.py`: EVALUATION.md settled on `held_out_comparison.py`.
- **Size and CI:**
  - "About 81% of sv_pgs is slated for removal" is now 74% (48,616 of 65,304 lines), because new modules landed.
  - The CI list is now 15 files, 8 of them old-path.
