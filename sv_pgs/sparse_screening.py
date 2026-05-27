"""Sparse marginal-z screening for rare variants.

This module provides an O(carrier_count) per-variant computation of the
marginal z-score used in the screening step, avoiding the O(n_samples)
cost of the dense formulation when the variant is rare (most genotypes
are zero).

Math derivation
---------------
Let g[:, j] be the raw genotype column for variant j (values in {0, 1, 2}),
and define the standardized column as

    x[i, j] = (g[i, j] - means[j]) / scales[j].

The dense marginal z is

    z_j = (x_j^T r) / sqrt((x_j^T x_j) * var(r))

where r is the current residual y - X_cov alpha.

Expand x_j^T r:
    x_j^T r = sum_i x[i, j] * r[i]
            = (1 / scales[j]) * sum_i g[i, j] * r[i]
              - (means[j] / scales[j]) * sum_i r[i]
            = (1 / scales[j]) * sum_{i in C_j} g[i, j] * r[i]
              - (means[j] / scales[j]) * sum_i r[i]

because g[i, j] is zero for i not in the carrier set C_j.

Expand x_j^T x_j:
    x_j^T x_j = sum_i ((g[i, j] - means[j]) / scales[j])^2
              = (1 / scales[j]^2) * (
                    sum_i g[i, j]^2
                    - 2 * means[j] * sum_i g[i, j]
                    + n_samples * means[j]^2
                )
              = (1 / scales[j]^2) * (
                    sum_{i in C_j} g[i, j]^2
                    - 2 * means[j] * sum_{i in C_j} g[i, j]
                    + n_samples * means[j]^2
                )

Again because g[i, j] is zero off-carriers, the inner sums collapse to
the carrier set, giving O(|C_j|) per variant.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def compute_sparse_marginal_z(
    carrier_index_per_variant: list[NDArray],
    carrier_genotype_per_variant: list[NDArray],
    means: NDArray,
    scales: NDArray,
    residual: NDArray,
    n_samples: int,
) -> NDArray:
    """Return marginal z-score for each rare variant.

    Equivalent to the dense formulation

        z_j = (x_j^T residual) / sqrt(x_j^T x_j * var(residual))

    with x_j = (raw_j - means[j]) / scales[j], but computed only over the
    carriers' contributions in O(|C_j|) per variant.

    Parameters
    ----------
    carrier_index_per_variant
        For each rare variant j, an int32 array of sample indices i with
        g[i, j] != 0.
    carrier_genotype_per_variant
        For each rare variant j, an int8 array of the corresponding
        genotype values (1 or 2), aligned with ``carrier_index_per_variant[j]``.
    means
        (n_rare,) per-variant raw-genotype means used for standardization.
    scales
        (n_rare,) per-variant scaling factors used for standardization.
    residual
        (n_samples,) current residual y - X_cov alpha.
    n_samples
        Number of samples (length of the dense column, including non-carriers).

    Returns
    -------
    NDArray
        (n_rare,) marginal z-scores.
    """
    n_rare = len(carrier_index_per_variant)
    if len(carrier_genotype_per_variant) != n_rare:
        raise ValueError(
            "carrier_index_per_variant and carrier_genotype_per_variant "
            "must have the same length"
        )
    if means.shape[0] != n_rare or scales.shape[0] != n_rare:
        raise ValueError("means and scales must have length n_rare")

    residual = np.asarray(residual, dtype=np.float64)
    sum_r = float(residual.sum())
    # var(r) using the same convention as dense screening: population variance.
    # (Either convention cancels as long as it matches dense; we use the
    # uncorrected mean of squared deviations.)
    var_r = float(residual.var())

    z = np.empty(n_rare, dtype=np.float64)

    for j in range(n_rare):
        idx = carrier_index_per_variant[j]
        gen = carrier_genotype_per_variant[j].astype(np.float64, copy=False)
        mu = float(means[j])
        sc = float(scales[j])

        if sc == 0.0:
            z[j] = 0.0
            continue

        if idx.size == 0:
            sum_gr = 0.0
            sum_g = 0.0
            sum_g2 = 0.0
        else:
            r_sub = residual[idx]
            sum_gr = float(np.dot(gen, r_sub))
            sum_g = float(gen.sum())
            sum_g2 = float(np.dot(gen, gen))

        # x_j^T r
        xtr = sum_gr / sc - (mu / sc) * sum_r

        # x_j^T x_j
        xtx = (sum_g2 - 2.0 * mu * sum_g + n_samples * mu * mu) / (sc * sc)

        denom = np.sqrt(xtx * var_r)
        if denom == 0.0:
            z[j] = 0.0
        else:
            z[j] = xtr / denom

    return z
