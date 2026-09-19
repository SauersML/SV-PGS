# The dosage store

One consolidated spec. It replaces the numbered addenda A4 through A4.15. Code: `dosage_store.py`, `store_converter.py`, `external_annotations.py`, `gatksv_source.py`, `gatksv_store_rows.py`, `synthetic_store.py`.

**Data class:** every store built from AoU data is participant-derived and stays inside the AoU workspace where it was built. Only counts-only QC with n ≥ 21 per cell leaves a task.

## Dosage arrays
- **Layout:** `dosage/half{h}/chrK` is a Zarr v3 uint8 array `[n_records, n_samples_h]`.
  - Rows are in popped-BCF file order, all records.
  - Columns are (batch, header position) only; no sample names are stored.
- **Code:** `(DS_milli·127 + 500) // 1000`, where DS_milli is the corrected dosage below. 255 is never written.
- **Encoding:**
  - shards of 65,536 rows with 64-row inner chunks;
  - zstd level 3, with a crc32c-checked shard index;
  - a `transcode_store` step builds an uncompressed local cache.
- **Kernels** read codes as signed `code − 127` and accumulate in int32, exactly.
- **Background removal ("value matched"),** applied before quantization and before any sums:
  - Per record, K_v = min(10, N_PATHS_TOTAL); m_v = the number of kept paths carrying the record's ID; w = ε/(1−ε) with the imputation error ε = 0.001.
  - The background values are q_v = m_v·w/(1 + K_v·w) and 2q_v. DS is set to 0 wherever its 3-dp value equals 0, q_v or 2q_v; every other value is left untouched.
  - In simulation this leaves zero background residual in every record type, and the only error is ≤ 2q_v, on carriers.
  - It is exact where a per-variant modal floor or a global 2ε is not, since the latter mishandle multi-path and PL-bearing records.
- **Recalibration:** where a validated per-stratum κ exists (from truth), the stored value is D* = μ + κ(DS − μ). It is linear only. Until the truth-derived κ table arrives, the field stays empty; there is no default.

## Halves: one verified site list, several measurements
- Every half shares the chromosome's site list and its sites md5. The fit gives each half its own covariate.
- MANIFEST `half_measurements[h]` names the measurement:
  - **imputed_dosage:** the two GLIMPSE2 imputation halves. Background is removed as above, and D* is applied where κ exists.
  - **long_read_calls:** the ~12k long-read panel members, as hard-call ALT counts on the same records. There is no background to remove, and the half is its own reliability class.
- **No-calls** occur only in hard-call halves. A no-call is filled with the ancestry group's measured mean at that record over every half. Where the group has no measurement, the record's pooled mean is used. A record measured in no sample is an error.

## Sidecar `variants/chrK` (one row per record)
- **Keys:** pos, ref_len, alt_len, refalt_md5, atomic IDs.
- **variant_class:** SNV → snv; INDEL → small_indel. For SVs, the first match wins:
  1. VNTR/STR context → str_vntr_repeat;
  2. INS/DUP with an MEI TE class → insertion_mei;
  3. DEL → deletion;
  4. DUP → duplication;
  5. otherwise other_complex_sv.
  - GATK-SV adds copy_number (multiallelic CNVs, read from FORMAT/CN) and inversion. Breakends are dropped.
  - Length is not a class; it enters the prior as a smooth of log length. A GATK-SV row's symbolic REF/ALT carry no length, so its sv_length annotation comes from |SVLEN| or the END span (`GatksvRows.lengths`).
  - Every variant_class column stores its legend, and a store whose legend differs from `VariantClass` is refused. Stores written before 2cc7aed, which merged the length-binned DEL/DUP classes (on MSI: s1M_100k, mini, build-lead/synth), must be converted again.
- **Grouping:** bubble_idx, same_pos_first.
- **Reliability keys:** sv_ctx, cx, has_pl.
- **r2_truth (f32):** corr²(stored D, G), triad-corrected, computed in-workspace from the r̂ model's coefficients. It stays NaN until they arrive.
- **tr_locus (u32):**
  - the connected component of GIAB v3.6 "AllTandemRepeatsandHomopolymers_slop5" intervals that the record's trimmed core overlaps;
  - a record bridging intervals merges them;
  - homopolymers are included, and there is no padding.
- **SV context** (`store_converter.sv_kernel_features`): the features of one learned distance kernel over every SV allele of the chromosome outside the record's own bubble. There is no window, no K and no frequency cutoff.
  - Each allele weighs 2f(1 − f), its genotype-variance share; this equals H_locus = 1 − Σ f_a² for a biallelic locus.
  - Nesting (the record's core inside the allele's) is its own feature per class: the sum of weights.
  - Every other allele adds weight × B_m(log(1 + gap)) per class, with the gap in bases between the cores and B_m cubic B-splines uniform in log(1 + gap) over [0, log(1 + the chromosome's extent)], once plain and once times log length.
  - The knot spacing starts at one per octave and is halved until the fit's evidence stops changing; the smoothness is learned, and the fit centres the raw sums (docs/design/math/scale_model.md).
  - Storage plan (not yet written by the converter):
    - features are kept at the finest spacing the fit will use, since coarser uniform B-splines on a dyadic grid are exact combinations of finer ones;
    - each feature is quantized to the store's 1/254 step with its own scale, which is within the prior's tolerance because features enter log u linearly;
    - near bases are sparse;
    - a class-specific block is kept where n · Var(feature) · τ̂² clears the certificate tolerance (τ̂² from hyperprior_pooling), and the class-summed block otherwise.
  - Pairs are exact, about 2e10 on chr1. The derived far-basis binning width b/d ≤ (8/3)·(1/254)·h is barely cheaper.
- **Per-half sums:** sum_code, sum_code2 and no_calls. They give AF, variance and rsq_ds.

## Loci `loci/chrK`
tr_start, tr_end, n_intervals, n_records, n_dlen_nonzero, tr_motif_len (the majority TRF period), and r2_locus (corr²(Z, true locus length), NaN until available).

## External annotations
- **Maps:**
  - `ext/<source>/rec_map/<chrom>`: matched records only, with rec_idx, ext_id, tier;
  - `ext/<source>/locus_map/<chrom>`: tr_locus → ext_locus_id, tier;
  - `ext/<source>/payload/<trait>`: ext_id → z².
  - An unmatched record has no row, so absence is explicit rather than zero-filled.
- **Tiers:** 1 exact allele, 2 sequence match, 3 coordinate-only.
- **Matching:** assignment is one-to-one greedy in both directions, by tier, quality, breakpoint distance, then ID.
  - VNTR loci join by overlap, with the external motif length (the trailing integer of the external locus ID) agreeing up to a 2× or 3× period.
  - The payload is z², so allele polarity cannot flip an annotation. A counts-only gate still flags pairs whose frequency fits the flipped allele better.
- **Sources:** Bai et al. 2026 SV and VNTR releases, and Pan-UKB EUR SNVs (the primary SNV side, lifted over to GRCh38 once for all traits).

## Gates (fail closed; only counts leave a task)
- **G1–G7:** inputs, sites, samples, FORMAT, write-back, and a cross-check of raw per-batch sums against the imputation pipeline's site statistics.
- **G8:** excess co-carriage over independence within a TR locus, tested only where the rarer record has ≥ 21 carrier haplotypes. It detects the same length change recorded twice.
- **Floor QC:** zeroed fractions by class × cx × has_pl.
- **Service-half gates, which must pass before its data is pooled:**
  - **S0, image identity: PASS.** GLIMPSE2 1.2.0-8671138 vs 1.0.0-2cee597. At one thread all 433,383 records of a public 50-sample chr22 shard are identical. At four threads, genotype discordance is 1.01–1.02× the same-image replicate floor. So one r² curve serves both halves.
  - **S1 sites, S2 FORMAT, S3 floor:** pending until that half's data arrives.

## MANIFEST
It records, per half, the measurement type, GLIMPSE2 image digest and pop binary md5; the BED names and md5s; the r² model version, target, covariates, coefficient md5 and CV error; each column's data class; and the gate results.
