"""Monte Carlo precision of an empirical rejection rate across independent replicates."""
from __future__ import annotations

import math
from statistics import NormalDist


def rejection_summary(successes: int, replicates: int) -> dict[str, float | int]:
    """Wilson 95% interval remains informative when no/all simulations reject."""
    if not 0 <= successes <= replicates:
        raise ValueError("successes must lie between zero and replicates")
    if replicates == 0:
        return dict(replicates=0, rate=math.nan, mcse=math.nan, lower=math.nan, upper=math.nan)
    rate = successes / replicates
    z = NormalDist().inv_cdf(0.975)
    denominator = 1.0 + z * z / replicates
    center = (rate + z * z / (2 * replicates)) / denominator
    half = z * math.sqrt(rate * (1 - rate) / replicates + z * z / (4 * replicates**2)) / denominator
    return dict(replicates=replicates, rate=rate, mcse=math.sqrt(rate * (1 - rate) / replicates),
                lower=max(0.0, center - half), upper=min(1.0, center + half))
