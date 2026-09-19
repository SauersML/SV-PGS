"""The result of a variational EB fit: posterior means and variances, hyperparameters, diagnostics."""
from __future__ import annotations

from dataclasses import dataclass, field

from sv_pgs._typing import NDArray
from sv_pgs.config import VariantClass


@dataclass(slots=True)
class VariationalFitResult:
    alpha: NDArray  # float32
    beta_reduced: NDArray  # float32
    beta_variance: NDArray  # float64 — preserves precision for shrunk variants
    prior_scales: NDArray  # float64 — preserves precision for shrunk variants
    global_scale: float
    class_tpb_shape_a: dict[VariantClass, float]
    class_tpb_shape_b: dict[VariantClass, float]
    scale_model_coefficients: NDArray
    scale_model_feature_names: list[str]
    sigma_error2: float
    objective_history: list[float]
    validation_history: list[float]
    member_prior_variances: NDArray  # float64 — preserves precision for shrunk variants
    linear_predictor: NDArray | None = None
    selected_iteration_count: int | None = None
    converged: bool = False
    final_parameter_change: float | None = None
    final_predictor_change: float | None = None
    final_objective_change: float | None = None
    final_hyperparameter_change: float | None = None
    elbo_history: list[float] = field(default_factory=list)
    # Intercept shift for the damped posterior predictive (predict_proba); 0.0
    # when posterior variances were not computed and predict_proba is the plug-in.
    predictive_intercept_shift: float = 0.0
