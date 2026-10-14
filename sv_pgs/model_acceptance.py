"""Numerical release gate for fitted research models, separate from deployment validation.

Refused trial steps are diagnostic, not automatically fatal. Unresolved returned states
are fatal. Saving a diagnostic artifact is allowed; scoring or reporting it is not.
"""
from __future__ import annotations

import numpy as np


def acceptance(model) -> dict:
    issues = []
    terms = model.certificate
    count = len(model.model_names)

    def issue(code, severity, detail):
        issues.append({"code": code, "severity": severity, "detail": detail})

    for key, required in (("outer_criterion_met", True),
                          ("budget_unresolved", False), ("restored_best", False), ("far_field_scale", False)):
        if key not in terms:
            issue("missing_" + key, "fatal", "Returned-state diagnostic is absent")
            continue
        values = np.asarray(terms[key])
        if values.shape != (count,) or not np.all(np.isfinite(values)) or np.any(values != int(required)):
            issue(key, "fatal", "At least one returned model failed this check")
    # This is a historical count of refused trials without a fixed point, not the
    # convergence status of the returned state (OuterFit.unresolved). Legitimate
    # failed line-search trials must not quarantine a later accepted model.
    if "unresolved" in terms:
        values = np.asarray(terms["unresolved"])
        if values.shape != (count,) or not np.all(np.isfinite(values)) or np.any(values < 0):
            issue("invalid_trial_count", "fatal", "Invalid unresolved-trial history")
        elif np.any(values):
            issue("unresolved_trials", "info", "Some refused trial states had no fixed point")
    for key in ("remaining_gain", "mean_error", "mean_move", "noise_gain", "prediction_move", "far_field"):
        if key in terms and not np.all(np.isfinite(terms[key])):
            issue(key, "fatal", "Non-finite returned-state diagnostic")
    # A fit on the band alone (a summary route, since deleted) left the far field out of its likelihood and recorded its
    # measured scale; every route records 0, so a nonzero scale is an artifact of that route, refused.
    if "far_field_scale" in terms and np.any(np.asarray(terms["far_field_scale"]) != 0):
        issue("truncated_ld", "fatal", "Summary-only LD approximation is not validated for primary results")
    for reason in model.refusals:
        fatal = "lattice check unresolved" in reason
        issue("quadrature_unresolved" if fatal else "fit_note", "fatal" if fatal else "info", reason)
    for scoring in model.scoring:
        for name in ("coefficients", "signed_means", "signed_scales", "alpha"):
            if not np.all(np.isfinite(getattr(scoring, name))):
                issue("nonfinite_" + name, "fatal", "Invalid scoring parameters")
        if np.any(np.asarray(scoring.signed_scales) <= 0):
            issue("invalid_scale", "fatal", "Scoring scales must be positive")
    issue("approximate_inference", "info", "Local numerical acceptance does not establish posterior coverage or an exact evidence bound")
    return {"schema": "svpgs-acceptance-v1", "accepted": not any(i["severity"] == "fatal" for i in issues),
            "deployment_validated": False, "issues": issues}


def require_accepted(model) -> None:
    decision = acceptance(model)
    if not decision["accepted"]:
        fatal = ", ".join(i["code"] for i in decision["issues"] if i["severity"] == "fatal")
        raise ValueError(f"Model quarantined from scoring and primary reporting: {fatal}")
