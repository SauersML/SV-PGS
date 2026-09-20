"""Closed-form accuracy and detection limits for the bench-tox and AoU designs (replica.polygenic_r2, detection_variance).

The inputs are design facts (n, p) and ranges for h^2 and M_e. The outputs are labelled as extrapolations.
"""
import numpy as np

from benchmarks.closed_form import replica

DESIGNS = {
    "bench-real per gene (n=585, p=3e4)": (585, 3e4),
    "bench-tox genome-wide (n=634 train, p=4e7)": (634, 4e7),
    "AoU half (n=25k, p=1e8)": (25_000, 1e8),
    "AoU full (n=50k, p=1e8)": (50_000, 1e8),
}
HERITABILITIES = (0.1, 0.3, 0.5)
EFFECTIVE_SEGMENTS = (6e4, 1.5e5)


def main():
    for name, (count, dimension) in DESIGNS.items():
        print(f"== {name}")
        print(f"   smallest recoverable single-effect variance share (k=1): {replica.detection_variance(dimension, count):.2e}; "
              f"(k=100): {replica.detection_variance(dimension, count, causal=100):.2e}")
        for effective in EFFECTIVE_SEGMENTS:
            values = ", ".join(f"h2={h:.1f}: {replica.polygenic_r2(count, h, effective):.4f}" for h in HERITABILITIES)
            print(f"   polygenic (Gaussian prior) r2 at M_e={effective:.0e}: {values}")


if __name__ == "__main__":
    main()
