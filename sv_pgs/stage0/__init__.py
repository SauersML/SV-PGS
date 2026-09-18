"""Stage 0: the single phenotype-independent genotype pass (exact per-LD-block integer Grams).

The CUDA backend lives in ``sv_pgs.stage0.cuda_backend`` and imports CuPy; everything
exported here runs on NumPy/SciPy alone.
"""

from __future__ import annotations

from sv_pgs.stage0.cpu_backend import CpuStage0Backend
from sv_pgs.stage0.genotype_pass import (
    GenotypePassPlan,
    GenotypePassSummary,
    GenotypeTileSource,
    Stage0Backend,
    assign_chromosomes,
    plan_genotype_pass,
    run_genotype_pass,
)
from sv_pgs.stage0.layout import SampleLayout, build_sample_layout
from sv_pgs.stage0.partition import OnlineBlockPartitioner, cut_allowed_from_groups
from sv_pgs.stage0.statistics import (
    BlockStatistics,
    block_correlation,
    centered_cross_products,
    column_moments,
    pooled_integer_statistics,
)

__all__ = [
    "BlockStatistics",
    "CpuStage0Backend",
    "GenotypePassPlan",
    "GenotypePassSummary",
    "GenotypeTileSource",
    "OnlineBlockPartitioner",
    "SampleLayout",
    "Stage0Backend",
    "assign_chromosomes",
    "block_correlation",
    "build_sample_layout",
    "centered_cross_products",
    "column_moments",
    "cut_allowed_from_groups",
    "plan_genotype_pass",
    "pooled_integer_statistics",
    "run_genotype_pass",
]
