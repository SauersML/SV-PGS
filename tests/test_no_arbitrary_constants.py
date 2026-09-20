"""SPEC: no hand-chosen constants anywhere in the package.

Every numeric literal in ``sv_pgs/`` is either exempt by its form or registered
below with the reason it is not a choice:

- exempt: 0, 1, -1, 2 and halves; indices and slices; square, cube and
  inverse exponents; ``axis=``-style arguments; values that only appear inside
  a formatted string (display, never results);
- registered: a specification or file format, a mathematical constant, a
  definition, a derivation, the configuration of an external run whose output
  we must match, a unit, or (in the simulator only) a scenario's parameter,
  each with its citation;
- pending: a known arbitrary constant with the lane that owns its removal. Each
  one still in the code is debt, listed by the strict xfail below until the
  last is gone.

Any other literal is an arbitrary constant: derive it, learn it, measure it at
runtime, or delete the feature that needed it. Removing a pending constant
never fails a test (drop its entry when convenient); a registered entry whose
value no longer occurs does fail, so the citations never outlive the code.
"""
from __future__ import annotations

import ast
from pathlib import Path

import pytest

PACKAGE = Path(__file__).resolve().parents[1] / "sv_pgs"

EXEMPT_VALUES = frozenset({0, 1, -1, 2, 0.5, -0.5})
EXEMPT_EXPONENTS = frozenset({2, 3, -1, -2, 0.5, -0.5, 1.5, -1.5})
EXEMPT_KEYWORDS = frozenset({"axis", "ndim", "ord", "stacklevel", "maxsplit", "indent"})

CLASSES = frozenset({"spec", "math", "definitional", "derived", "external-config", "unit", "scenario"})
# A scenario parameter defines a simulated truth, never a method's choice, so only the simulator may register one.
SCENARIO_MODULES = frozenset({"synthetic_store.py"})

# (module path relative to sv_pgs/, symbol) -> (class, allowed values, citation)
REGISTRY: dict[tuple[str, str], tuple[str, frozenset[object], str]] = {
    ('all_of_us.py', 'DISEASE_DEFINITIONS.lab_criteria'): ('spec', frozenset({6.5, 60.0, 90, 30.0}), 'ADA Standards of Care (HbA1c >= 6.5%); KDIGO 2012 CKD (GFR < 60 or ACR >= 30 mg/g for > 90 days)'),
    ('all_of_us.py', 'DISEASE_DEFINITIONS.minimum_case_age_years'): ('spec', frozenset({40.0}), 'GOLD report: COPD diagnosis from age 40'),
    ('all_of_us.py', 'PREGNANCY_WINDOW_DAYS_BEFORE'): ('spec', frozenset({294}), '42 weeks, where post-term begins; ACOG Committee Opinion 736 (2018): postpartum through 12 weeks'),
    ('all_of_us.py', 'PREGNANCY_WINDOW_DAYS_AFTER'): ('spec', frozenset({294, 84}), '42 weeks, where post-term begins; ACOG Committee Opinion 736 (2018): postpartum through 12 weeks'),
    ('all_of_us.py', 'SELF_REPORT_TYPE_CONCEPT_ID'): ('spec', frozenset({32865}), 'OMOP / All of Us PPI concept ids'),
    ('all_of_us.py', 'MEASUREMENT_DEFINITIONS.physical_measurement_concept_ids'): ('spec', frozenset({903133, 903118, 903126}), 'OMOP / All of Us PPI concept ids'),
    ('all_of_us.py', 'MEASUREMENT_DEFINITIONS.unit_conversions'): ('unit', frozenset({2.54000508, 2.54, 100.0, 30.48, 30.480061, 0.09148, 2.152}), 'inch, foot and US survey inch/foot to cm; creatinine 1 mg/mmol = 8.84 mg/g; NGSP-IFCC master equation (ngsp.org)'),
    ('all_of_us.py', 'MEASUREMENT_DEFINITIONS.minimum_age_years'): ('spec', frozenset({20.0}), 'CDC 2000 growth charts end at age 20'),
    ('all_of_us.py', 'MEASUREMENT_DEFINITIONS.excluded_source_concept_ids'): ('spec', frozenset({903109, 903114, 903130, 903112, 903105, 903108}), 'OMOP / All of Us PPI concept ids'),
    ('all_of_us.py', 'LAB_ANALYTE_DEFINITIONS.unit_conversions'): ('unit', frozenset({8.84}), 'inch, foot and US survey inch/foot to cm; creatinine 1 mg/mmol = 8.84 mg/g; NGSP-IFCC master equation (ngsp.org)'),
    ('all_of_us.py', 'CKD_EPI_2021_SEX_COEFFICIENTS'): ('spec', frozenset({0.7, -0.241, 1.012, 0.9, -0.302}), 'CKD-EPI 2021 creatinine equation (Inker et al. 2021, NEJM 385:1737)'),
    ('all_of_us.py', 'build_all_of_us_measurement_targets'): ('definitional', frozenset({0.25, 0.75}), 'quartiles of a metadata summary'),
    ('all_of_us.py', 'MINIMUM_REPORTED_PARTICIPANTS'): ('spec', frozenset({21}), 'All of Us Data and Statistics Dissemination Policy: no count of 1 to 20'),
    ('all_of_us.py', '_age_on'): ('derived', frozenset({7, 365.25}), 'only the birth year is released, so July 1 (month 7) is the unbiased birthday; 365.25-day Julian year'),
    ('cli.py', 'main'): ('spec', frozenset({130}), 'POSIX exit status 128 + SIGINT'),
    ('code_products.py', 'DIGIT_BITS'): ('derived', frozenset({7}), 'the widest balanced digit that fits int8 with room for the sign'),
    ('code_products.py', 'OPERAND_DIGITS'): ('derived', frozenset({8}), '7 x 8 - 2 >= 53 fp64 mantissa bits (math-epeb)'),
    ('code_products.py', 'INT32_EXACT_DIGIT_ROWS'): ('definitional', frozenset({31}), 'int32 range 2^31 - 1'),
    ('code_products.py', '_FLOAT64_BYTES'): ('derived', frozenset({8}), "byte accounting of this module's arrays: itemsizes and the number of simultaneously live arrays"),
    ('code_products.py', '_INT32_BYTES'): ('derived', frozenset({4}), "byte accounting of this module's arrays: itemsizes and the number of simultaneously live arrays"),
    ('exact_marginals_scale.py', 'FORMATION_ROUNDINGS'): ('definitional', frozenset({5}), 'roundings in one term D_j xt_ij xt_kj beyond the accumulation depth: the reciprocal (once), sqrt(D_j) in both factors, the two scalings; Higham 2002 Lemma 3.1'),
    ('exact_marginals_scale.py', 'VARIANCE_OPERATIONS'): ('definitional', frozenset({5}), 'roundings in D (1 - D v): the reciprocal D = 1 / Pi entering twice, the product, the difference and the scaling; Higham 2002 section 3.1'),
    ('exact_marginals_scale.py', 'exact_dual_cost'): ('definitional', frozenset({3}), 'Cholesky and triangular-inverse flop counts n^3/3; Golub & Van Loan 2013 section 4.2'),
    ('code_products.py', '_digit_working_bytes'): ('derived', frozenset({4}), "byte accounting of this module's arrays: itemsizes and the number of simultaneously live arrays"),
    ('code_products.py', 'INT8_GEMM_ALIGNMENT'): ('spec', frozenset({4}), 'cuBLAS int8 GEMM leading dimensions must be multiples of 4'),
    ('code_products.py', 'CodeBlockTile._codes_times'): ('derived', frozenset({3}), "byte accounting of this module's arrays: itemsizes and the number of simultaneously live arrays"),
    ('code_products.py', 'CodeBlockTile._codes_times_operand'): ('derived', frozenset({3}), "byte accounting of this module's arrays: itemsizes and the number of simultaneously live arrays"),
    ('code_products.py', 'CodeBlockTile._squared_codes_times_operand'): ('derived', frozenset({4, 3}), "byte accounting of this module's arrays: itemsizes and the number of simultaneously live arrays"),
    ('code_products.py', 'CodeBlockTile._codes_times_weighted_codes'): ('derived', frozenset({3}), "byte accounting of this module's arrays: itemsizes and the number of simultaneously live arrays"),
    ('cohort.py', 'SECOND_DEGREE_KINSHIP'): ('spec', frozenset({-3.5}), 'KING second-degree kinship bound 2^-3.5 (Manichaikul et al. 2010, Table 1)'),
    ('compute_budget.py', '_CGROUP_V1_UNLIMITED'): ('spec', frozenset({63}), 'Linux PAGE_COUNTER_MAX, (2^63 - 1) rounded down to a page (include/linux/page_counter.h)'),
    ('compute_budget.py', '_detect_available_host_ram_bytes'): ('unit', frozenset({1024}), '/proc/meminfo reports kB'),
    ('dosage_store.py', 'CODES_PER_DOSAGE'): ('spec', frozenset({127}), '8-bit store format (STORE.md); a CRC32C is 4 bytes'),
    ('dosage_store.py', 'MAXIMUM_CODE'): ('spec', frozenset({254}), '8-bit store format (STORE.md); a CRC32C is 4 bytes'),
    ('dosage_store.py', 'MISSING_CODE'): ('spec', frozenset({255}), '8-bit store format (STORE.md); a CRC32C is 4 bytes'),
    ('dosage_store.py', 'MAXIMUM_DOSAGE_MILLI'): ('spec', frozenset({2000}), '8-bit store format (STORE.md); a CRC32C is 4 bytes'),
    ('dosage_store.py', 'ZSTD_LEVEL'): ('spec', frozenset({3}), 'libzstd ZSTD_CLEVEL_DEFAULT'),
    ('dosage_store.py', '_UNWRITTEN_CHUNK'): ('definitional', frozenset({64}), 'uint64 maximum as the unwritten-chunk sentinel'),
    ('dosage_store.py', '_SHARD_INDEX_ENTRY_BYTES'): ('spec', frozenset({16}), '8-bit store format (STORE.md); a CRC32C is 4 bytes'),
    ('dosage_store.py', '_CRC32C_BYTES'): ('spec', frozenset({4}), '8-bit store format (STORE.md); a CRC32C is 4 bytes'),
    ('dosage_store.py', 'encode_dosage_milli'): ('definitional', frozenset({500, 1000}), 'round-half-up integer division in milli units'),
    ('dosage_store.py', '_code_array_metadata'): ('spec', frozenset({3}), 'Zarr format version 3'),
    ('dosage_store.py', '_layout_from_metadata'): ('spec', frozenset({3}), 'Zarr format version 3'),
    ('dosage_store.py', '_column_metadata'): ('spec', frozenset({3}), 'Zarr format version 3'),
    ('dosage_store.py', 'open_column'): ('spec', frozenset({3}), 'Zarr format version 3'),
    ('dosage_store.py', 'chromosome_number'): ('spec', frozenset({22}), 'human autosomes chr1-chr22'),
    ('marginal_variances.py', 'exact_route_is_cheaper'): ('spec', frozenset({9.0, 3.0}), 'leading-order flop counts: symmetric eigendecomposition with vectors about 9 n^3, Cholesky n^3 / 3 (Golub and Van Loan, Matrix Computations, 4th ed., Table 8.3.1 and Section 4.2)'),
    ('dosage_store.py', 'DEFAULT_INNER_CHUNK_ROWS'): ('derived', frozenset({64}), 'minimax-regret chunk rows over read runs no shorter than the smallest Stage 0 tile, from the measured read-path cost fit (docs/design/math/codec.md §3)'),
    ('dual_solve.py', 'DIGIT_BITS'): ('derived', frozenset({7}), 'the widest balanced digit that fits int8 with room for the sign'),
    ('engine_kernels.py', '_tilted_row_bytes'): ('derived', frozenset({3}), "byte accounting of a chunk's device copies: its three float64 inputs (log u, P, h)"),
    ('engine_kernels.py', '_objective_row_bytes'): ('derived', frozenset({6}), "byte accounting of a chunk's device arrays: three float64 inputs, two outputs and the |log Z| temporary per row"),
    ('resident_codes.py', 'read_tile_rows'): ('derived', frozenset({3}), "byte accounting of CodeBlockTile._codes_times: the fp64 total and two recombination terms per row"),
    ('external_annotations.py', 'TIER_COORDINATE'): ('definitional', frozenset({3}), 'enum code; (chromosome << 32 | position) site key'),
    ('external_annotations.py', 'MOTIF_MULTIPLES'): ('spec', frozenset({3}), 'TRF reports a period, or twice or three times it'),
    ('external_annotations.py', '_SITE_STRIDE'): ('definitional', frozenset({32}), 'enum code; (chromosome << 32 | position) site key'),
    ('fold_share.py', 'sharing_cost'): ('math', frozenset({3.0}), 'Cholesky n^3/3 and Householder QR 2mn^2 - 2n^3/3 flop counts (Golub and Van Loan, Matrix Computations, 4th ed., sections 4.2.5 and 5.2.2)'),
    ('fast_scoring.py', 'SIGNED_CODE_OFFSET'): ('spec', frozenset({127.0}), '8-bit store code offset (STORE.md)'),
    ('fast_scoring.py', '_FLOAT64_BYTES'): ('definitional', frozenset({8}), 'float64 itemsize'),
    ('fast_scoring.py', 'predictive_intercept_shift.xtol'): ('derived', frozenset({4.0}), "rtol = 4 eps is scipy's minimum; xtol = 4 eps min(p, 1 - p) since the logistic slope is at most 1/4"),
    ('fast_scoring.py', 'predictive_intercept_shift.rtol'): ('derived', frozenset({4.0}), "rtol = 4 eps is scipy's minimum; xtol = 4 eps min(p, 1 - p) since the logistic slope is at most 1/4"),
    ('logistic_ep.py', '_LOG_SIGMOID_CURVATURE_BOUND'): ('math', frozenset({0.25}), 'max of sigmoid(w) sigmoid(-w), the curvature of -log sigmoid, attained at w = 0'),
    ('genotype_buffers.py', 'SIGNED_CODE_OFFSET'): ('spec', frozenset({127}), '8-bit store code offset (STORE.md)'),
    ('genotype_buffers.py', 'build_sample_layout'): ('definitional', frozenset({3}), 'a sample correlation needs at least three samples to be non-degenerate'),
    ('genotype_buffers.py', 'host_buffer_bytes'): ('derived', frozenset({4, 16, 3, 8}), "byte accounting of this module's allocations: 4-byte int32/float32, 8-byte int64/float64, two int64 row sums, three block matrices"),
    ('genotype_buffers.py', 'cuda_buffer_bytes'): ('derived', frozenset({16, 8, 4}), "byte accounting of this module's allocations: 4-byte int32/float32, 8-byte int64/float64, two int64 row sums, three block matrices"),
    ('genotype_buffers.py', '_CUDA_R_8I'): ('spec', frozenset({3}), 'cudaDataType and cublasComputeType_t values (library_types.h, cublas_api.h)'),
    ('genotype_buffers.py', '_CUDA_R_32I'): ('spec', frozenset({10}), 'cudaDataType and cublasComputeType_t values (library_types.h, cublas_api.h)'),
    ('genotype_buffers.py', '_CUBLAS_COMPUTE_32I'): ('spec', frozenset({72}), 'cudaDataType and cublasComputeType_t values (library_types.h, cublas_api.h)'),
    ('genotype_buffers.py', 'CudaGenotypeBuffer._gemm'): ('derived', frozenset({4}), "byte accounting of this module's allocations: 4-byte int32/float32, 8-byte int64/float64, two int64 row sums, three block matrices"),
    ('genotype_statistics.py', 'TIE_CORRELATION_SCREEN'): ('derived', frozenset({6.0}), '1 - gamma_6: six fp64 roundings after an exact integer (Higham, Accuracy and Stability, gamma_n)'),
    ('held_out_comparison.py', 'size_gate(alpha=)'): ('spec', frozenset({0.05}), 'pre-registered test level (EVALUATION.md G13)'),
    ('held_out_comparison.py', 'null_cost_gate(alpha=)'): ('spec', frozenset({0.05}), 'pre-registered test level (EVALUATION.md G13)'),
    ('imputation_reliability.py', 'triad_squared_correlation'): ('definitional', frozenset({3}), 'a correlation needs at least three observations'),
    ('ld_partition.py', '_UNREACHABLE'): ('definitional', frozenset({61}), 'a quarter of the int64 range: a sum of two reachable costs stays representable'),
    ('marginal_variances.py', '_edgeworth_correction'): ('math', frozenset({6.0}), 'one-term Edgeworth polynomial of the studentized mean, q1(x) = g (2 x^2 + 1) / 6 (Hall 1992, The Bootstrap and Edgeworth Expansion, section 2.6)'),
    ('phenotype_measurement.py', 'FIT_TOLERANCE'): ('derived', frozenset({4}), 'searches compare spans of four evidence values to EVIDENCE_TOLERANCE; fits short of their maxima by a quarter of it move a span by at most half'),
    ('phenotype_measurement.py', 'level_posterior'): ('math', frozenset({0.25}), 'Var_k(e^2 / (2 s_k)) = e^4 Var_k(1 / s_k) / 4, the noise score variance of the split information'),
    ('phenotype_measurement.py', '_SPLIT_OPERATIONS_PER_TERM'): ('derived', frozenset({8}), "the rounded operations behind each (node, occasion, component) term of the split information, counted in its comment"),
    ('phenotype_measurement.py', '_LATTICE_MATRICES'): ('derived', frozenset({8}), "the noise lattice's K x K float64 matrices live at once, counted in its comment"),
    ('phenotype_measurement.py', '_GOLDEN'): ('math', frozenset({3.0, 5.0}), 'golden-section ratio (3 - sqrt 5) / 2'),
    ('phenotype_measurement.py', '_FLOAT64_BYTES'): ('definitional', frozenset({8}), 'float64 itemsize'),
    ('phenotype_measurement.py', '_ENTRY_BYTES'): ('derived', frozenset({4}), "byte accounting of the E-step: four float64 arrays per (person, node, occasion, component) entry live at once"),
    ('phenotype_measurement.py', '_Model.__init__'): ('math', frozenset({12.0}), 'variance of a uniform rounding error on a width-delta grid, delta^2 / 12 (Sheppard)'),
    ('phenotype_measurement.py', '_level_grid'): ('derived', frozenset({0.25}), 'the relative tolerance split: half to the trapezoid rule, a quarter to each truncated tail'),
    ('phenotype_measurement.py', '_log_prior_sum_bound'): ('derived', frozenset({3.0, 4.0}), 'moments m = 0, 1, 2; |T|^m N(T; 0, tau^2) has four monotone pieces for m > 0 (two for m = 0)'),
    ('progress.py', 'elapsed'): ('unit', frozenset({60}), 'seconds per minute and minutes per hour'),
    ('scale_mixture_ep.py', 'ROUGHNESS_ORDER'): ('spec', frozenset({3}), 'third-difference roughness (lead ruling 6c9976a): its null space is a normal log-density, the proper lambda = infinity limit'),
    ('scale_mixture_ep.py', '_ROW_INTERMEDIATES'): ('derived', frozenset({20}), "byte accounting: the (rows x K) float64 arrays alive at once in a chunk"),
    ('scale_mixture_ep.py', '_QUADPACK_RELATIVE_FLOOR'): ('spec', frozenset({50.0}), 'QUADPACK / scipy.integrate.quad: epsrel must exceed 50 x machine epsilon'),
    ('scale_mixture_ep.py', 'kernel_floor'): ('derived', frozenset({0.25, 4.0}), '|log L| <= v|h^2 - P|/2 + (vP)^2/4, the second-order bound; its root in the stable form 2c/(b + sqrt(b^2 + 4ac))'),
    ('scale_mixture_ep.py', 'tail_mass'): ('math', frozenset({-2.0}), 'Gaussian integral by completing the square: erfcx form of the half-line integral'),
    ('scale_mixture_ep.py', '_legendre_functionals'): ('definitional', frozenset({3}), 'a quadratic (the Legendre P0..P2 span) needs three nodes'),
    ('scale_mixture_ep.py', '_components.third'): ('derived', frozenset({6.0}), 'the recursion A_(n+1) = r A_n - r(1 - r) A_n\', B_(n+1) = -r(1 - r) B_n\' for d^n log Z_k / d eta^n (docstring)'),
    ('scale_mixture_ep.py', '_components.fourth'): ('derived', frozenset({3.0, 9.0, 6.0, -3.0}), 'the recursion A_(n+1) = r A_n - r(1 - r) A_n\', B_(n+1) = -r(1 - r) B_n\' for d^n log Z_k / d eta^n (docstring)'),
    ('scale_mixture_ep.py', 'quadrature_majorant_ratio'): ('math', frozenset({0.25}), '|(1 + i q)^(-1/2)| = (1 + q^2)^(-1/4) at t + i pi/2'),
    ('scale_mixture_ep.py', '_directional_derivatives'): ('math', frozenset({4, 3.0, 6.0, 4.0}), 'derivatives of a log-sum-exp as joint cumulants (Faa di Bruno): k4 = E[X^4] - 3 Var^2, d4 = k4 + 6 k(X, X, Y) + 3 Var(Y) + 4 Cov(X, Z) + E W'),
    ('scale_mixture_ep.py', '_variant_derivatives'): ('math', frozenset({3.0}), 'd Var / dh of a normal mixture: E[(mu - m)^3] + 3 E[c (mu - m)] (third central moment of a Gaussian mixture)'),
    ('scale_mixture_ep.py', '_stationarity_check'): ('derived', frozenset({3.0, 4.0, 0.75, 0.25}), 'central difference with V certified to e: error h^2 s/6 + e/h, least at h = (3e/s)^(1/3) where it is (3^(2/3)/2) s^(1/3) e^(2/3); e from 1/2 E^2/s = tolerance/(4n), i.e. e = (2 tolerance/(n 3^(4/3)))^(3/4) s^(1/4)'),
    ('scale_mixture_ep.py', '_stationarity_check.bound'): ('derived', frozenset({6.0}), 'the central difference truncation h^2 s / 6 (Taylor remainder of V with |V\'\'\'| <= s)'),
    ('scale_mixture_ep.py', '_laplace_corrections'): ('math', frozenset({8.0, 5.0, 24.0}), 'Tierney and Kadane (1986): the O(1) Laplace term E[u^4]/24 k4 + E[u^6]/72 k3^2 = k4/8 + 5 k3^2/24'),
    ('store_converter.py', 'IMPUTATION_ERROR_RATE'): ('external-config', frozenset({0.001}), 'GLIMPSE2 --err-imp of the aou2_50k imputation run; the removed background is an exact function of it'),
    ('store_converter.py', 'MAXIMUM_KEPT_PATHS'): ('external-config', frozenset({10}), 'pop-glimpse2 max_alleles of the aou2_50k imputation run'),
    ('store_converter.py', 'NO_LOCUS'): ('definitional', frozenset({4294967295}), 'uint32 maximum as the no-locus sentinel'),
    ('store_converter.py', 'DECODE_STEP_BYTES_PER_ENTRY'): ('derived', frozenset({5, 4, 10}), "per-entry byte bounds of the converter's steps (itemsizes of their arrays)"),
    ('store_converter.py', 'ASSEMBLY_STEP_BYTES_PER_ENTRY'): ('derived', frozenset({4, 10, 3}), "per-entry byte bounds of the converter's steps (itemsizes of their arrays)"),
    ('store_converter.py', 'value_matched_background'): ('definitional', frozenset({1000.0, 2000.0}), 'milli units of a diploid dosage'),
    ('store_converter.py', 'KERNEL_STEP_BYTES_PER_PAIR'): ('derived', frozenset({3, 6}), "per-entry byte bounds of the converter's steps (itemsizes of their arrays)"),
    ('store_converter.py', '_uniform_cubic_weights'): ('math', frozenset({6.0, 3.0, 4.0, -3.0}), 'uniform cubic B-spline basis polynomials (divided by 6)'),
    ('store_converter.py', 'sv_kernel_features'): ('derived', frozenset({4}), 'a cubic B-spline basis has intervals + degree + 1 functions'),
    ('store_converter.py', 'refalt_digest'): ('definitional', frozenset({16}), 'a 64-bit digest is 16 hex digits'),
    ('store_converter.py', '_imputed_dosage_milli'): ('definitional', frozenset({1000.0}), 'milli units of a diploid dosage'),
    ('sv_fusion.py', 'MAXIMUM_BREAKPOINT_DISTANCE'): ('spec', frozenset({100}), 'GATK-SV re-clustering rule'),
    ('sv_fusion.py', 'MINIMUM_CALIBRATION_SAMPLES'): ('math', frozenset({4}), 'Fisher z has standard error 1/sqrt(n - 3), so n > 3'),
    ('sv_fusion.py', '_fisher_z'): ('math', frozenset({3}), 'Fisher z has standard error 1/sqrt(n - 3), so n > 3'),
    ('sv_fusion.py', 'mean_anchor'): ('derived', frozenset({4.0}), 'delta-method variance of the intercept; 4 = 2^2 from the diploid dosage scale'),
    ('synthetic_store.py', 'HG38_AUTOSOME_MEGABASES'): ('scenario', frozenset({248.96, 242.19, 198.3, 190.21, 181.54, 170.81, 159.35, 145.14, 138.39, 133.8, 135.09, 133.28, 114.36, 107.04, 101.99, 90.34, 83.26, 80.37, 58.62, 64.44, 46.71, 50.82}), 'simulator scenario parameters, with the units, code ranges and RNG stream labels they use (synthetic_store module docstring)'),
    ('synthetic_store.py', 'COHORT_GROUPS'): ('scenario', frozenset({0.8, 0.2, 0.35, 0.15}), 'simulator scenario parameters, with the units, code ranges and RNG stream labels they use (synthetic_store module docstring)'),
    ('synthetic_store.py', 'ANCESTRY_DIRICHLET_CONCENTRATION'): ('scenario', frozenset({15.0}), 'simulator scenario parameters, with the units, code ranges and RNG stream labels they use (synthetic_store module docstring)'),
    ('synthetic_store.py', 'ADMIXTURE_GENERATIONS'): ('scenario', frozenset({8}), 'simulator scenario parameters, with the units, code ranges and RNG stream labels they use (synthetic_store module docstring)'),
    ('synthetic_store.py', 'TILE_GAP_BP'): ('scenario', frozenset({1000000}), 'simulator scenario parameters, with the units, code ranges and RNG stream labels they use (synthetic_store module docstring)'),
    ('synthetic_store.py', 'TILE_GAP_CM'): ('scenario', frozenset({50.0}), 'simulator scenario parameters, with the units, code ranges and RNG stream labels they use (synthetic_store module docstring)'),
    ('synthetic_store.py', 'ERR_IMP'): ('scenario', frozenset({0.001}), 'simulator scenario parameters, with the units, code ranges and RNG stream labels they use (synthetic_store module docstring)'),
    ('synthetic_store.py', 'POP_CLAMP'): ('scenario', frozenset({1e-05}), 'simulator scenario parameters, with the units, code ranges and RNG stream labels they use (synthetic_store module docstring)'),
    ('synthetic_store.py', 'POP_KEPT_PATHS'): ('scenario', frozenset({10}), 'simulator scenario parameters, with the units, code ranges and RNG stream labels they use (synthetic_store module docstring)'),
    ('synthetic_store.py', 'NESTED_RECORD_PROBABILITY'): ('scenario', frozenset({0.3}), 'simulator scenario parameters, with the units, code ranges and RNG stream labels they use (synthetic_store module docstring)'),
    ('synthetic_store.py', 'MAXIMUM_NESTED_PATHS'): ('scenario', frozenset({5}), 'simulator scenario parameters, with the units, code ranges and RNG stream labels they use (synthetic_store module docstring)'),
    ('synthetic_store.py', 'SOFT_POSTERIOR_FRACTION'): ('scenario', frozenset({0.1}), 'simulator scenario parameters, with the units, code ranges and RNG stream labels they use (synthetic_store module docstring)'),
    ('synthetic_store.py', 'SOFT_DELTA_LOG10_RANGE'): ('scenario', frozenset({-5.0, -2.0}), 'simulator scenario parameters, with the units, code ranges and RNG stream labels they use (synthetic_store module docstring)'),
    ('synthetic_store.py', 'SINGLE_PATH_R2'): ('scenario', frozenset({0.8329993319973283, 0.9543537402545237, 0.9917454154040785, 0.9979868123448445, 0.726935085825916, 0.9306600470124883, 0.9865243070521974, 0.9967590549910882, 0.0022485654501617743, 0.1738066716577618, 0.6865351451998423, 0.868419151645502, 0.06496154204265829, 0.30317890563719757, 0.6548260520832868, 0.9006202645418034}), 'bench-sim v7 Beagle 5.5 arm median single-record dosage r2 on public 1kGP haplotypes (v7/results_calibration_beagle_5000.json)'),
    ('synthetic_store.py', 'MAF_BIN_EDGES'): ('scenario', frozenset({0.001, 0.01, 0.05}), 'the MAF bin edges of bench-sim v7 Beagle 5.5 arm median single-record dosage r2 on public 1kGP haplotypes (v7/results_calibration_beagle_5000.json)'),
    ('synthetic_store.py', 'PIPELINE_R2_LOSS'): ('scenario', frozenset({0.01}), 'stated design input: half B\'s r2 target sits 0.01 below half A\'s for every class; no measured or AoU source (synthetic_store module comment)'),
    ('synthetic_store.py', 'R2_BETA_CONCENTRATION'): ('scenario', frozenset({20.0}), 'simulator scenario parameters, with the units, code ranges and RNG stream labels they use (synthetic_store module docstring)'),
    ('synthetic_store.py', 'COMPLEXITY_BIN_PATHS'): ('scenario', frozenset({5, 6, 10, 11, 20, 21, 60}), 'simulator scenario parameters, with the units, code ranges and RNG stream labels they use (synthetic_store module docstring)'),
    ('synthetic_store.py', '_tile_units'): ('scenario', frozenset({3}), 'simulator scenario parameters, with the units, code ranges and RNG stream labels they use (synthetic_store module docstring)'),
    ('synthetic_store.py', '_popcount'): ('scenario', frozenset({8}), 'simulator scenario parameters, with the units, code ranges and RNG stream labels they use (synthetic_store module docstring)'),
    ('synthetic_store.py', '_SOFT_DELTA_MEAN'): ('scenario', frozenset({10}), 'simulator scenario parameters, with the units, code ranges and RNG stream labels they use (synthetic_store module docstring)'),
    ('synthetic_store.py', '_SOFT_DELTA_SECOND_MOMENT'): ('scenario', frozenset({10}), 'simulator scenario parameters, with the units, code ranges and RNG stream labels they use (synthetic_store module docstring)'),
    ('synthetic_store.py', 'noise_parameters'): ('scenario', frozenset({10}), 'simulator scenario parameters, with the units, code ranges and RNG stream labels they use (synthetic_store module docstring)'),
    ('synthetic_store.py', 'draw_tile_mosaic'): ('scenario', frozenset({100.0}), 'simulator scenario parameters, with the units, code ranges and RNG stream labels they use (synthetic_store module docstring)'),
    ('synthetic_store.py', 'haplotype_posteriors'): ('scenario', frozenset({10.0}), 'simulator scenario parameters, with the units, code ranges and RNG stream labels they use (synthetic_store module docstring)'),
    ('synthetic_store.py', 'half_statistics'): ('scenario', frozenset({1000.0, 256, 1000000.0, 500}), 'simulator scenario parameters, with the units, code ranges and RNG stream labels they use (synthetic_store module docstring)'),
    ('synthetic_store.py', 'plan_store'): ('scenario', frozenset({3}), 'simulator scenario parameters, with the units, code ranges and RNG stream labels they use (synthetic_store module docstring)'),
    ('synthetic_store.py', '_write_variant_table'): ('scenario', frozenset({31}), 'simulator scenario parameters, with the units, code ranges and RNG stream labels they use (synthetic_store module docstring)'),
    ('variant_typing.py', 'COMMUNITY_SV_MINIMUM_LENGTH'): ('spec', frozenset({50}), 'community SV definition, >= 50 bp; reporting label and external-record matching only'),
    ('variant_typing.py', '_core_kind_and_length'): ('external-config', frozenset({10.0, 0.1}), 'imputation strata near-pure rule max(10, 10%) (lrma-strata scripts/finalqc/strata_build.py classify)'),
}

# (module path relative to sv_pgs/, symbol) -> (owner lane, values, what replaces them)
PENDING: dict[tuple[str, str], tuple[str, frozenset[object], str]] = {
    ('scale_mixture_ep.py', '_maximize_coefficients'): ('e2e', frozenset({0.25, 0.75}), 'the trust-region gain-ratio thresholds and radius factors are the illustrative defaults of Algorithm 4.1 (any eta in [0, 1/4) converges): set the radius from the accuracy of the model itself (the cubic-term bound) instead'),
    ('scale_mixture_ep.py', '_ascend_evidence'): ('e2e', frozenset({0.25, 0.75}), 'the trust-region gain-ratio thresholds and radius factors are the illustrative defaults of Algorithm 4.1 (any eta in [0, 1/4) converges): set the radius from the accuracy of the model itself (the cubic-term bound) instead'),
    ('all_of_us.py', 'MIN_PRE_LANDMARK_CONDITION_DATES'): ('phenotypes', frozenset({5}), 'latent-class EHR evidence model (PHENOTYPES.md item 1)'),
    ('all_of_us.py', 'EHR_DEPTH_LANDMARK_DAYS'): ('phenotypes', frozenset({365}), 'latent-class EHR evidence model (PHENOTYPES.md item 1)'),
    ('all_of_us.py', 'DISEASE_DEFINITIONS.minimum_control_age_years'): ('phenotypes', frozenset({40.0, 50.0}), 'latent-class EHR evidence model (PHENOTYPES.md item 1)'),
    ('all_of_us.py', 'ADULT_AGE_YEARS'): ('phenotypes', frozenset({18}), 'per-occasion measurement model with learned time-since-event smooths and a contamination component (PHENOTYPES.md)'),
    ('all_of_us.py', 'ACUTE_CARE_WINDOW_DAYS'): ('phenotypes', frozenset({30}), 'per-occasion measurement model with learned time-since-event smooths and a contamination component (PHENOTYPES.md)'),
    ('all_of_us.py', 'HEMATOLOGIC_MALIGNANCY'): ('phenotypes', frozenset({180}), 'per-occasion measurement model with learned time-since-event smooths and a contamination component (PHENOTYPES.md)'),
    ('all_of_us.py', 'CHEMOTHERAPY'): ('phenotypes', frozenset({90}), 'per-occasion measurement model with learned time-since-event smooths and a contamination component (PHENOTYPES.md)'),
    ('all_of_us.py', 'TRANSFUSION'): ('phenotypes', frozenset({120}), 'per-occasion measurement model with learned time-since-event smooths and a contamination component (PHENOTYPES.md)'),
    ('all_of_us.py', 'ANTIBACTERIAL_COURSE'): ('phenotypes', frozenset({7, 14}), 'per-occasion measurement model with learned time-since-event smooths and a contamination component (PHENOTYPES.md)'),
    ('all_of_us.py', 'ACUTE_HEPATOBILIARY'): ('phenotypes', frozenset({30}), 'per-occasion measurement model with learned time-since-event smooths and a contamination component (PHENOTYPES.md)'),
    ('all_of_us.py', 'DISEASE_DEFINITIONS.lab_criteria.plausible_range'): ('phenotypes', frozenset({3.0, 20.0, 0.2, 0.1, 30000.0}), 'kept on the lab-criterion path only (lead ruling): the disease measurement model (PHENOTYPES.md item 1, part 2) replaces it with a learned unit-confusion component'),
    ('all_of_us.py', 'MEASUREMENT_DEFINITIONS.treatment'): ('phenotypes', frozenset({15.0, 0.7}), 'within-person learned treatment effect (PHENOTYPES.md item 3)'),
    ('config.py', 'ModelConfig.minimum_scale'): ('e2e', frozenset({1e-06}), 'Stage 0 inclusion by the information rule and an exact zero-variance test, not a MAF or SD floor'),
    ('config.py', 'ModelConfig.minimum_minor_allele_frequency'): ('e2e', frozenset({0.01}), 'Stage 0 inclusion by the information rule and an exact zero-variance test, not a MAF or SD floor'),
    ('config.py', 'ModelConfig.prior_scale_floor'): ('e2e', frozenset({1e-06}), "B7: the engine's EB replaces the fixed prior-scale clamps"),
    ('config.py', 'ModelConfig.prior_scale_ceiling'): ('e2e', frozenset({10.0}), "B7: the engine's EB replaces the fixed prior-scale clamps"),
    ('config.py', 'ModelConfig.local_scale_floor'): ('e2e', frozenset({1e-08}), "B7: the engine's EB replaces the fixed prior-scale clamps"),
    ('data.py', 'VariantRecord.allele_frequency'): ('deslop-constants', frozenset({0.01}), 'require allele_frequency instead of a silent 0.01 default'),
    ('data.py', 'normalize_variant_record.allele_frequency'): ('deslop-constants', frozenset({0.01}), 'require allele_frequency instead of a silent 0.01 default'),
    ('dosage_store.py', 'DEFAULT_SHARD_ROWS'): ('speed-io', frozenset({65536}), 'read time does not depend on it (codec.md §3); derive from file and descriptor count against parallel shard writers'),
    ('exact_polish.py', '_MAXIMUM_CONJUGATE_GRADIENT_ITERATIONS'): ('deslop-fit', frozenset({400}), 'stop on the CG A-norm error estimate or the fp64 floor (math-epeb); the dimension bound is exact'),
    ('exact_polish.py', '_DIRECTION_DROP'): ('deslop-fit', frozenset({1e-20}), 'block-CG deflation at eps x block norm'),
    ('exact_polish.py', '_DRAW_BLOCK_COLUMNS'): ('deslop-fit', frozenset({64}), 'measure from the device budget'),
    ('exact_polish.py', '_MINIMUM_NEWTON_STEP'): ('deslop-fit', frozenset({-30}), 'stop on the Newton decrement at fp64 resolution; test separation first (math-epeb)'),
    ('exact_polish.py', '_COVARIATE_NEWTON_ITERATIONS'): ('deslop-fit', frozenset({64}), 'stop on the Newton decrement at fp64 resolution; test separation first (math-epeb)'),
    ('exact_polish.py', '_COVARIATE_NEWTON_TOLERANCE'): ('deslop-fit', frozenset({1e-12}), 'stop on the Newton decrement at fp64 resolution; test separation first (math-epeb)'),
    ('exact_polish.py', '_COVARIATE_RIDGE'): ('deslop-fit', frozenset({1e-10}), 'covariates are projected exactly (FWL): detect rank deficiency exactly instead of a ridge'),
    ('external_annotations.py', 'KMER_LENGTH'): ('deslop-store', frozenset({11}), 'match tiers bin continuous evidence; store similarity, ratio and proximity as continuous columns (lead/engine ruling)'),
    ('fast_scoring.py', '_READ_AHEAD_BUFFERS'): ('speed-io', frozenset({3}), 'I/O concurrency from a measurement'),
    ('genotype_buffers.py', 'LAYOUT_ALIGNMENT'): ('speed-io', frozenset({64}), 'cite the cuBLAS int8 tensor-core alignment requirement, or derive from it'),
    ('genotype_buffers.py', 'PROFILE_SAMPLE_TARGET'): ('speed-io', frozenset({16384}), 'derive the cut-profile subsample from the cost model, or use every sample'),
    ('genotype_buffers.py', '_ROW_ALIGNMENT'): ('speed-io', frozenset({16}), 'cite the cuBLAS int8 tensor-core alignment requirement, or derive from it'),
    ('genotype_buffers.py', '_HOST_PANEL'): ('speed-io', frozenset({512}), 'tile and CUDA block sizes from cache and device attributes'),
    ('genotype_buffers.py', '_HOST_CROSS_ROWS'): ('speed-io', frozenset({64}), 'tile and CUDA block sizes from cache and device attributes'),
    ('genotype_buffers.py', '_CROSS_SAMPLE_CHUNK'): ('speed-io', frozenset({8192}), 'tile and CUDA block sizes from cache and device attributes'),
    ('genotype_buffers.py', '_CUDA_GRAM_PANEL'): ('speed-io', frozenset({512}), 'tile and CUDA block sizes from cache and device attributes'),
    ('genotype_buffers.py', 'CudaGenotypeBuffer.load_staged_tile'): ('speed-io', frozenset({256}), 'tile and CUDA block sizes from cache and device attributes'),
    ('genotype_buffers.py', 'CudaGenotypeBuffer.pair_weights'): ('speed-io', frozenset({256}), 'tile and CUDA block sizes from cache and device attributes'),
    ('genotype_statistics.py', 'BLOCK_CAP_STEP'): ('speed-floor', frozenset({256}), 'block-cap search step and tile floors from the budget and alignment'),
    ('genotype_statistics.py', 'plan_genotype_pass'): ('speed-floor', frozenset({64}), 'block-cap search step and tile floors from the budget and alignment'),
    ('genotype_statistics.py', 'plan_genotype_pass.capacity_rows'): ('speed-floor', frozenset({3}), 'block-cap search step and tile floors from the budget and alignment'),
    ('hyperprior_pooling.py', '_VARIANCE_SHRINK_LIMIT'): ('deslop-constants', frozenset({1e-06}), 'exact omega^2 = 0 boundary by the variance-component score test (math-epeb)'),
    ('hyperprior_pooling.py', '_QUADRATIC_FLOOR'): ('deslop-constants', frozenset({1e-12}), 'exact nu = infinity boundary by the T_k statistic (math-density)'),
    ('ld_partition.py', 'PAIR_WEIGHT_SCALE'): ('deslop-constants', frozenset({20}), "largest fixed-point scale the int64 cost range allows for the chromosome's size and block cap"),
    ('prior_design.py', '_continuous_spline_knots'): ('e2e', frozenset({0.25, 0.75}), 'a rich basis with a learned roughness penalty instead of quartile knots'),
    ('prior_design.py', '_effective_prior_variances'): ('e2e', frozenset({1e-08}), "B7: the engine's EB replaces the floor"),
    ('sv_fusion.py', 'MINIMUM_PAIRING_Z'): ('novel-measure', frozenset({5.0}), 'fusion acceptance: one event vs two in LD needs a prior on r2_B, not a z or clamp threshold'),
    ('sv_fusion.py', 'MAXIMUM_SECOND_RELIABILITY'): ('novel-measure', frozenset({1.25}), 'fusion acceptance: one event vs two in LD needs a prior on r2_B, not a z or clamp threshold'),
}


def _literal(node: ast.AST) -> object | None:
    if isinstance(node, ast.Constant) and type(node.value) in (int, float, complex):
        return node.value
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and isinstance(node.operand, ast.Constant)
        and type(node.operand.value) in (int, float, complex)
    ):
        return -node.operand.value
    return None


class _LiteralFinder(ast.NodeVisitor):
    """Collects (line, symbol, value) for every literal not exempt by its form.

    A literal's symbol is its module- or class-level assignment target, its
    function's qualified name (``name(argument=)`` for a default), or ``<module>``,
    extended by the keyword it is passed as (``TARGET.keyword``).
    """

    def __init__(self) -> None:
        self.found: list[tuple[int, str, object]] = []
        self.scope: list[str] = []
        self.function_depth = 0
        self.symbol: str | None = None

    def _visit_as(self, node: ast.AST, symbol: str | None) -> None:
        saved = self.symbol
        self.symbol = symbol
        self.visit(node)
        self.symbol = saved

    def visit_FunctionDef(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        qualified = ".".join([*self.scope, node.name])
        arguments = node.args
        positional = arguments.posonlyargs + arguments.args
        for argument, default in zip(positional[len(positional) - len(arguments.defaults) :], arguments.defaults):
            self._visit_as(default, f"{qualified}({argument.arg}=)")
        for argument, default in zip(arguments.kwonlyargs, arguments.kw_defaults):
            if default is not None:
                self._visit_as(default, f"{qualified}({argument.arg}=)")
        for decorator in node.decorator_list:
            self.visit(decorator)
        self.scope.append(node.name)
        self.function_depth += 1
        for statement in node.body:
            self._visit_as(statement, qualified)
        self.function_depth -= 1
        self.scope.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for decorator in node.decorator_list:
            self.visit(decorator)
        self.scope.append(node.name)
        for statement in node.body:
            self.visit(statement)
        self.scope.pop()

    def _assigned(self, targets: list[ast.expr], value: ast.AST) -> None:
        if self.function_depth:
            self.visit(value)
        else:
            self._visit_as(value, ".".join([*self.scope, ",".join(ast.unparse(target) for target in targets)]))

    def visit_Assign(self, node: ast.Assign) -> None:
        self._assigned(node.targets, node.value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        if node.value is not None:
            self._assigned([node.target], node.value)

    def visit_JoinedStr(self, node: ast.JoinedStr) -> None:
        return

    def visit_Subscript(self, node: ast.Subscript) -> None:
        self.visit(node.value)

    def visit_BinOp(self, node: ast.BinOp) -> None:
        self.visit(node.left)
        if isinstance(node.op, ast.Pow) and _literal(node.right) in EXEMPT_EXPONENTS:
            return
        self.visit(node.right)

    def visit_keyword(self, node: ast.keyword) -> None:
        if node.arg is None:
            self.visit(node.value)
        elif node.arg not in EXEMPT_KEYWORDS:
            self._visit_as(node.value, f"{self.symbol or '.'.join(self.scope) or '<module>'}.{node.arg}")

    def _record(self, node: ast.expr) -> None:
        value = _literal(node)
        if value not in EXEMPT_VALUES:
            self.found.append((node.lineno, self.symbol or ".".join(self.scope) or "<module>", value))

    def visit_Constant(self, node: ast.Constant) -> None:
        if _literal(node) is not None:
            self._record(node)

    def visit_UnaryOp(self, node: ast.UnaryOp) -> None:
        if _literal(node) is None:
            self.generic_visit(node)
        else:
            self._record(node)


def _scanned_modules() -> list[Path]:
    return sorted(PACKAGE.rglob("*.py"))


def _literals() -> list[tuple[str, int, str, object]]:
    rows = []
    for path in _scanned_modules():
        finder = _LiteralFinder()
        finder.visit(ast.parse(path.read_text()))
        relative = str(path.relative_to(PACKAGE))
        rows.extend((relative, line, symbol, value) for line, symbol, value in finder.found)
    return rows


def _allowed(table: dict[tuple[str, str], tuple[str, frozenset[object], str]], module: str, symbol: str) -> frozenset[object]:
    return table.get((module, symbol), ("", frozenset(), ""))[1]


def test_every_numeric_literal_in_the_package_is_exempt_registered_or_pending() -> None:
    arbitrary = [
        f"sv_pgs/{module}:{line} {symbol} = {value!r}"
        for module, line, symbol, value in _literals()
        if value not in _allowed(REGISTRY, module, symbol) and value not in _allowed(PENDING, module, symbol)
    ]
    assert not arbitrary, (
        "arbitrary constants (derive, learn, measure at runtime, or delete the feature; register only a spec, "
        "math, definition, derivation, external-run setting or unit, with its citation):\n  " + "\n  ".join(arbitrary)
    )


def _present() -> dict[tuple[str, str], set[object]]:
    present: dict[tuple[str, str], set[object]] = {}
    for module, _, symbol, value in _literals():
        present.setdefault((module, symbol), set()).add(value)
    return present


def test_every_registered_entry_is_classified_and_still_occurs() -> None:
    present = _present()
    for key, (category, values, citation) in REGISTRY.items():
        assert category in CLASSES, f"{key}: unknown class {category!r}"
        assert category != "scenario" or key[0] in SCENARIO_MODULES, f"{key}: only the simulator has scenario parameters"
        assert citation, f"{key}: a registered constant needs its citation"
        assert key not in PENDING, f"{key}: registered and pending at once"
        missing = set(values) - present.get(key, set())
        assert not missing, f"{key}: registered values {sorted(map(repr, missing))} no longer occur; drop them"


@pytest.mark.xfail(strict=True, reason="pending arbitrary constants remain; each names the lane removing it")
def test_no_arbitrary_constant_is_pending() -> None:
    present = _present()
    remaining = {key: entry for key, entry in PENDING.items() if set(entry[1]) & present.get(key, set())}
    assert not remaining, "\n  ".join(f"sv_pgs/{module} {symbol}: {owner}, {plan}" for (module, symbol), (owner, _, plan) in remaining.items())
