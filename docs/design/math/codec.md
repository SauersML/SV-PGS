# The store's GPU-decodable codec (rowdict) and its chunk size

Code: `sv_pgs/rowdict_codec.py`, the `rowdict` chain in `sv_pgs/dosage_store.py`, and the sweep in `benchmarks/store_codec.py`. Numbers are labelled [semi-real]: they come from public bench-sim v7 chr22 codes (Beagle 5.5 imputation of 1kGP haplotype mosaics, 20,000 samples), never AoU.

## 1. Format
An inner chunk of R records of n codes is `[R × u32 frame sizes][R row frames]`, and the chain appends its crc32c. A row frame is:
- `[u8 k]`;
- the row's 2^k most frequent code values;
- one k-bit dictionary slot per sample, bit-packed little-endian;
- the E exception samples (uint16 when n ≤ 2^16, else uint32), then their E codes.

E is implied by the frame size, so no field repeats another. An exception's slot holds 0, and its code overwrites that value after the slots are decoded.

**Choosing k.** The row's size in bytes at depth k is

  S_k = 1 + 2^k + ⌈nk/8⌉ + (w + 1)·(n − c_k),

where c_k is the number of samples covered by the 2^k most frequent values and w is the sample width. The encoder stores the exact argmin over k = 0..8. There is no threshold; k = 0 is the sparse row (its mode plus carriers).

**Decoding.** A k ≤ 8 bit slot spans at most two bytes. Every decoder reads a 16-bit window at bit s·k, and the chunk's crc always follows its last frame.
- **GPU:** one block row per record. The dictionary is staged in shared memory, then one thread per sample decodes; a second kernel scatters the exceptions. Every output byte is written once from the slots, then once more per exception.
- **CPU:** `decode_rows`, the reference decoder, is the same arithmetic in numpy.

## 2. When rowdict beats zstd (the tier rule)
Take a tier that delivers B bytes/s of stored data to a host with C cores, feeding a GPU at PCIe rate P. Let b be the stored bits per code. The time per decoded code is then:
- **zstd:** max(b_z/8B, 1/(C·D_z)), where D_z is one core's zstd decode rate;
- **raw:** max(1/B, 1/P);
- **rowdict:** max(b_r/8B, τ_h·b_r/8, b_r/8P, 1/D_g), where τ_h is the host's per-stored-byte time (read, crc32c, locate) and D_g is the GPU decode rate.

rowdict beats zstd once b_r/8B < 1/(C·D_z), that is once B > b_r·C·D_z/8, and it beats raw whenever the tier or PCIe is the bound.

**Measured [semi-real].** The data are the first 65,536 chr22 records × 20,000 samples, on an A40 host with 8 task CPUs, from warm page cache, with one reader thread. Rates are decoded GB/s for requests of L = 4096 records (`benchmarks/store_codec.py` at 7f34651; JSON at `/scratch.global/sauer354/svpgs-team/codec/store_codec_v7_final.json`).

| array, R = 64 | stored bits/code | CPU read | to device, end to end | kernels alone |
|---|---|---|---|---|
| raw | 8 | 2.4 | 7.3 (preadv to pinned, then copy) | — |
| zstd(3) | 0.361 | 2.03 | — (host decode, then a raw copy) | — |
| rowdict | 0.499 | 0.52 (reference decoder) | **51.9** | 266 |

On all of chr22's 590,623 records, rowdict takes 0.674 bits/code and zstd 0.458 [semi-real]. rowdict is about 1.4× zstd's bytes, but decoding is 26× faster per host thread, and it is 16× smaller than raw.

**Break-even.** With b_r ≈ 0.5 and D_z ≈ 2.0 GB/s, B* ≈ 0.13·C GB/s, about 2.0 GB/s at C = 16.

**The rule:**
- a tier slower than B* (the bucket, network storage) stays zstd;
- the local cache of a GPU host (NVMe or RAM) is rowdict;
- a CPU-only host keeps raw, because the CPU decoder is a reference, not a fast path.

## 3. The chunk size R, from the read path
A consumer reads contiguous runs of L records:
- Stage 0 reads tiles of `tile_rows`;
- Stage 2 reads each LD block's span.

A run touches ⌈(L + R − 1)/R⌉ chunks on average over its alignment. Its host time is

  T(R; L) = τ·ŝ·(L + R − 1) + t_c·(L + R − 1)/R + t_r,

where:
- ŝ is the stored bytes per record (its frame plus its size entry);
- τ is the host's seconds per stored byte (preadv, crc32c, frame location);
- t_c is the fixed cost per chunk (the crc32c call and index entry);
- t_r is the fixed cost per request.

T is convex in R. Setting dT/dR = τŝ − t_c(L − 1)/R² to 0 gives

  R* = √((L − 1)·t_c/(τ·ŝ)).

For every request length, the chosen R's excess over T(R*; L) is at most the table's worst regret.

**Measured fit [semi-real].** The fit is the least-squares host share of `read_rows_to_device` over R ∈ {1, 4, …, 1024} × L ∈ {64, 512, 4096} (maximum relative residual 0.20; the task shared its node):
- τ = 2.0·10⁻¹⁰ s/byte (about 5 GB/s of stored bytes per thread);
- t_c = 1.18 µs per chunk;
- t_r = 110 µs per request;
- ŝ = 1,247 bytes at n = 20,000.

So t_c/(τŝ) = 4.7, and R* = 17, 49 and 139 at L = 64, 512 and 4096.

**Choosing R.** The request length is not known when the store is written. Stage 0 tiles are at least 64 records (`plan_genotype_pass`), and Stage 2 spans whole LD blocks. So R is taken as the divisor of the shard rows that minimizes the worst regret max_L T(R; L)/T(R*; L) − 1 over L ≥ 64, including the L → ∞ limit t_c/(Rτŝ).

| R | 1 | 4 | 16 | 32 | **64** | 128 | 256 | 1024 |
|---|---|---|---|---|---|---|---|---|
| worst regret, % | 474 | 119 | 30 | 15 | **7.4** | 18 | 41 | 182 |

The sweep agrees. R = 64 has the fastest host share at L = 64 and is within 3% of the fastest at L = 512 and 4096. zstd is fastest at R = 64 for L = 64, and within 5% of its fastest for every L.

**What this fixes.** It fixes `DEFAULT_INNER_CHUNK_ROWS = 64` for every chain as a derived value. It does not fix the shard rows: a reader pays a shard's fixed cost (open, index read, crc) once per process, so read time does not depend on the shard rows while the index stays negligible (16 B per 64 records). That constant stays pending with speed-io. What it trades is the file and descriptor count against how many writers can build one array's shards in parallel.

## 4. Result
- **Lossless:** exact round trips are tested at every depth 0..8, with n not a multiple of 4 or 8 and both sample widths. The GPU decode matches the CPU reference (`tests/test_rowdict_codec*.py`). Corruption fails the chunk crc32c or the size-table checks.
- **The remaining gap:** the host share is about 67–75 GB/s decoded per process against 266 GB/s for the kernels. It does not scale with threads: the per-chunk crc32c and the numpy frame location hold the GIL (1 thread: 66.6 GB/s; 8 threads: 74.6 GB/s).
  - The next steps are a device crc32c and device-side frame location from the size tables. Either one leaves the host a single preadv per request.
- **Wiring:** Stage 0/2 consumers still take host codes. `DosageStore.read_codes_to_device` is the entry point for a rowdict cache; moving `StoreBlockSource` onto it is the consumer-side change.

## 5. Reading only a fit's active rows (layer A)
A Stage 2 read needs the reduced model's rows, not the whole span between a block's first and last rows. A rowdict frame depends only on its record's codes and n, not on where it sits. So `read_rows_to_device(start, stop, out, decoder, rows=)` works like this:
- it reads only the inner chunks that hold a wanted row, with one read per run of consecutive chunks;
- it checks their crc32c and locates only the wanted frames;
- it decodes them, in order, into a compact target of `len(rows)` rows.

The host still reads the unwanted frames that share a chunk with a wanted one. With R = 64 and the fraction f of rows wanted spread at random, a chunk is skipped with probability (1 − f)^64. So the host reads essentially the whole span unless the wanted rows cluster. The device work and memory follow the wanted rows only.

**Staging.** Chunks move in windows of whole chunks, through one pinned buffer and one device buffer of max(largest chunk, min(encoded bytes, decoded bytes of `rows`)). Staging therefore never exceeds the decoded rows it serves, and a whole-chromosome read no longer pins the chromosome's encoded bytes at once.

**A fit-local copy of the reduced rows (layer B)** would also remove the host's unwanted bytes. It is built only when the fit's own numbers say it pays. There are P Stage 2 passes; the span's encoded bytes are E_s and the reduced rows' E_r (so E_r ≤ E_s); reading costs τ_r per byte and writing τ_w. The copy pays when

  P·(E_s − E_r)·τ_r > E_s·τ_r + E_r·τ_w,

since it is built by copying frames (a read of the span plus a write of the rows, with nothing re-encoded). With every record active, E_s − E_r is the tied records alone. That is to be measured on an all-active fit before B is built.
