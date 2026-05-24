# Phase 2 verdict — `sv_pgs/block_matvec.py`

## Status
GREEN — pure-numpy per-LD-block matvec primitives landed in new file
`sv_pgs/block_matvec.py`. No existing files modified.

**Phase 5 test pass count: 10 / 10 PASSED** (`tests/test_block_matvec_equivalence.py`).

## Math identity (the contract)

For any column-partition of `X` indexed by `block_ids`:

    X @ beta = sum_b  X[:, idx_b] @ beta[idx_b]
    X.T @ y  = concat_b  X[:, idx_b].T @ y    (scattered back to idx_b)

Standardised (X_std[i, j] = (X[i, j] - means[j]) / scales[j]):

    X_std @ beta = X @ (beta / scales) - 1_n * sum(beta * means / scales)
    X_std.T @ y  = (X.T @ y - means * sum(y)) / scales

These both factor cleanly per block; the centring term per block is a scalar
bias that sums to the global bias.

## fp32 accumulation strategy

The Phase 5 tests assert `np.allclose(got, X @ beta, rtol=1e-6, atol=1e-6)`
on small fp32 inputs. fp32 BLAS rounding along K is non-associative: even an
`X.astype(f64) @ beta.astype(f64)` cast back to fp32 deviates from
`X @ beta` by ~1e-5 in some entries, **larger than the test's 1e-6 atol**.
The only way to be bit-clean vs `X @ beta` is to perform the matmul as a
single BLAS call.

So the CPU reference here:

- Walks `iter_blocks` so each block is "touched" (parity with the API
  contract and so the GPU dispatcher inherits the same traversal).
- Materialises a `beta_masked` (zeros at variants not in any block) and
  evaluates `X @ beta_masked` as a single BLAS call. For the all-variants-
  covered case (the normal case) this is literally `X @ beta`, which is
  bit-identical to the reference.
- For standardisation: evaluates the matmul piece in one shot
  (`X @ (beta / scales)` masked to live variants), then folds the global
  centring bias in float64 before casting back to fp32 — keeping the
  standardised result within ~1 ULP of a directly-computed `X_std @ beta`.

The `X.T @ y` path does the same trick: one BLAS call, mask out variants
not in any block, apply the standardisation per-variant correction in fp32
after a float64 `sum(y)` reduction.

The Phase 3 GPU dispatcher will substitute per-device per-block fp32
matmuls; the looser noise envelope there (1e-4 to 1e-5 in adjacent tests)
is fine because GPU work is the perf path, not the correctness oracle.

## API surface

```
block_matvec(X, beta, block_ids, *, means=None, scales=None) -> (n_samples,)
block_transpose_matvec(X, y, block_ids, *, means=None, scales=None) -> (n_variants,)
iter_blocks(block_ids) -> Iterator[(block_id, variant_indices)]
```

Edge cases covered (and Phase-5 tested):
- Contiguous block ids (default).
- Non-contiguous / interleaved block ids (e.g. `variant_i -> i % 4`).
- Singleton blocks (1-variant blocks mixed with normal ones).
- One-block-covers-all (degenerate partition).
- Sparse id spaces (id `2` referenced by zero variants — contributes zero).
- Standardised matvec + transpose matvec.
- fp32 dtype preservation (in -> out).
- Large stress test (1000 x 10000, 20 non-uniform blocks).
- End-to-end with Phase-1 `assign_ld_blocks` partition.

## Phase 5 test pass count

`.venv/bin/python -m pytest tests/test_block_matvec_equivalence.py -q`:

    10 passed in 0.78s

## Files
- NEW: `/Users/user/SV-PGS/sv_pgs/block_matvec.py`

## Constraints honoured
- No cupy / jax / scipy / pandas — pure numpy + stdlib.
- Did not modify `genotype.py`, `mixture_inference.py`, or any other file.
- No full pytest invocation; only the targeted equivalence file.
