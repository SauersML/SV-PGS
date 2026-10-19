"""A fitted model's certification verdict, and what scoring it needs.

``acceptance`` certifies a returned state: its outer criterion met, no unresolved returned state, no open quadrature
check. A model that misses certification is still scored and reported, since held-out accuracy measures it whatever its
convergence, and the report lists it with its codes. ``require_scorable`` refuses only a model that cannot be scored:
non-finite or non-positive scoring parameters, or a group whose fit failed and returned its covariates alone.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def subset_refusals(refusals: Sequence[str], indices: Sequence[int]) -> tuple[str, ...]:
    """Select and renumber the model-indexed refusal notes written by ``stage2_wiring.fit_models``."""
    renumbered = {index: position for position, index in enumerate(indices)}
    selected = []
    for refusal in refusals:
        label, separator, reason = refusal.partition(": ")
        if not separator or not label.startswith("model "):
            raise ValueError(f"Fit refusal has no model index: {refusal}")
        index = int(label.removeprefix("model "))
        if index in renumbered:
            selected.append(f"model {renumbered[index]}: {reason}")
    return tuple(selected)


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
        if ": the fit failed: " in reason:
            issue("fit_failed", "fatal", reason)
        elif "lattice check unresolved" in reason:
            issue("quadrature_unresolved", "fatal", reason)
        else:
            issue("fit_note", "info", reason)
    for scoring in model.scoring:
        for name in ("coefficients", "signed_means", "signed_scales", "alpha"):
            if not np.all(np.isfinite(getattr(scoring, name))):
                issue("nonfinite_" + name, "fatal", "Invalid scoring parameters")
        if np.any(np.asarray(scoring.signed_scales) <= 0):
            issue("invalid_scale", "fatal", "Scoring scales must be positive")
    issue("approximate_inference", "info", "Local numerical acceptance does not establish posterior coverage or an exact evidence bound")
    return {"schema": "svpgs-acceptance-v1", "accepted": not any(i["severity"] == "fatal" for i in issues),
            "deployment_validated": False, "issues": issues}


UNSCORABLE = frozenset({"fit_failed", "nonfinite_coefficients", "nonfinite_signed_means", "nonfinite_signed_scales",
                        "nonfinite_alpha", "invalid_scale"})
"""The fatal codes under which a model has no score: every other fatal code leaves it uncertified but scored."""


def unscorable_codes(decision: dict) -> list[str]:
    """The fatal codes of an ``acceptance`` verdict that stop the model being scored."""
    return sorted({i["code"] for i in decision["issues"] if i["severity"] == "fatal" and i["code"] in UNSCORABLE})


def require_scorable(model) -> None:
    codes = unscorable_codes(acceptance(model))
    if codes:
        raise ValueError(f"Model cannot be scored: {', '.join(codes)}")


def require_accepted(model) -> None:
    decision = acceptance(model)
    if not decision["accepted"]:
        fatal = ", ".join(i["code"] for i in decision["issues"] if i["severity"] == "fatal")
        raise ValueError(f"Model quarantined from scoring and primary reporting: {fatal}")
