"""Pin: hardcall tie canonicalization detects complement-flipped pairs.

For int8 PLINK-style hardcalls (0/1/2 with -1 missing), two columns that are
arithmetic complements (col_b == 2 - col_a) standardize into mutual negations.
The tie-detection path canonicalizes such pairs by collapsing them into the
same group with sign = -1, saving downstream linear-algebra work.

Adversarial: construct two columns that are exact complements, confirm the
``sign_flipped`` canonical signature of one equals the ``exact`` canonical
signature of the other.

Also pin: two columns that are NEITHER equal NOR complements must NOT
collapse — the sign-flipped LUT lookup must NOT generate matching signatures.
"""
from __future__ import annotations

import numpy as np

from sv_pgs.plink import PLINK_MISSING_INT8
from sv_pgs.preprocessing import (
    _canonicalize_hardcall_tie_columns_i8,
    _hardcall_state_masks,
)


def test_complement_columns_match_under_sign_flip_canonical_form():
    """col_a = [0,1,2,0,1,2], col_b = [2,1,0,2,1,0] → exact(a) == sign_flipped(b)."""
    col_a = np.asarray([0, 1, 2, 0, 1, 2], dtype=np.int8)
    col_b = np.asarray([2, 1, 0, 2, 1, 0], dtype=np.int8)
    batch = np.stack([col_a, col_b], axis=1)  # shape (n_samples, 2)
    masks = _hardcall_state_masks(batch)
    exact, sign_flipped = _canonicalize_hardcall_tie_columns_i8(batch, masks)

    # The signature contract: column a's exact signature equals column b's
    # sign-flipped signature (and vice versa) when they are true complements.
    # Note: the LUT operates per-column based on each column's observed
    # state mask. Pinning what's observed:
    assert exact.shape == (6, 2)
    assert sign_flipped.shape == (6, 2)
    # If a and b are observed-pair complements with the same state mask,
    # then exact(a) == sign_flipped(b) elementwise (modulo encoding).
    assert np.array_equal(exact[:, 0], sign_flipped[:, 1]), (
        f"exact(col_a)={exact[:, 0]} should equal sign_flipped(col_b)={sign_flipped[:, 1]}"
    )
    assert np.array_equal(exact[:, 1], sign_flipped[:, 0]), (
        "complement detection is asymmetric; sign-flipped LUT not inverting"
    )


def test_identical_columns_match_under_exact_form():
    """Sanity: two identical columns have identical exact signatures."""
    col = np.asarray([0, 1, 2, 0, 1, 2], dtype=np.int8)
    batch = np.stack([col, col.copy()], axis=1)
    masks = _hardcall_state_masks(batch)
    exact, _ = _canonicalize_hardcall_tie_columns_i8(batch, masks)
    assert np.array_equal(exact[:, 0], exact[:, 1])


def test_unrelated_columns_do_not_match_under_sign_flip():
    """Two random non-complement columns: their exact and sign-flipped
    signatures must not coincide (no spurious tie)."""
    col_a = np.asarray([0, 0, 1, 2, 0, 1], dtype=np.int8)
    col_b = np.asarray([2, 1, 0, 0, 2, 0], dtype=np.int8)
    # Verify the precondition: b != 2 - a anywhere.
    assert not np.array_equal(col_b, (2 - col_a).astype(np.int8))
    batch = np.stack([col_a, col_b], axis=1)
    masks = _hardcall_state_masks(batch)
    exact, sign_flipped = _canonicalize_hardcall_tie_columns_i8(batch, masks)
    # Neither orientation should report a match between a and b.
    assert not np.array_equal(exact[:, 0], exact[:, 1])
    assert not np.array_equal(exact[:, 0], sign_flipped[:, 1])
    assert not np.array_equal(sign_flipped[:, 0], exact[:, 1])


def test_missing_values_propagate_through_canonical_form():
    """Missing entries (-1) must be marked as missing in the canonical form
    (encoded value 3 in the LUT)."""
    col_a = np.asarray([0, PLINK_MISSING_INT8, 2, 1], dtype=np.int8)
    batch = col_a[:, None]
    masks = _hardcall_state_masks(batch)
    exact, sign_flipped = _canonicalize_hardcall_tie_columns_i8(batch, masks)
    # The missing position must produce the same canonical value in both
    # exact and sign-flipped forms (because missing is order-invariant).
    assert exact[1, 0] == sign_flipped[1, 0]
