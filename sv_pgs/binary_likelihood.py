"""Binary traits: the Bernoulli likelihood by Polya-Gamma augmentation, its variational bound, and calibration.

**Model.** A case/control trait has labels y_i in {0, 1} on its training rows and the linear predictor
eta_i = c_i'alpha + x_i'beta (the covariates alpha under a flat prior, the effects beta under the learned prior), with
P(y_i = 1 | eta_i) = sigmoid(eta_i). With kappa_i = y_i - 1/2, the Polya-Gamma identity (Polson, Scott and Windle 2013,
JASA 108:1339, Theorem 1) writes each factor as a Gaussian scale mixture in eta_i,

    sigmoid(eta)^y sigmoid(-eta)^(1 - y) = 2^-1 e^(kappa eta) E_omega[e^(-omega eta^2 / 2)],   omega ~ PG(1, 0).

**The bound.** A variational factor q(omega_i) = PG(1, xi_i) gives, after the omega integral, the bound of Jaakkola and
Jordan (2000, Statistics and Computing 10:25) in its Polya-Gamma form (Durante and Rigon 2019, Statistical Science 34:472):

    log p(y_i | eta_i) >= kappa_i eta_i - omega_i eta_i^2 / 2 + c(xi_i),
    omega_i = E[omega_i] = tanh(xi_i / 2) / (2 xi_i)   (1/4 at xi_i = 0, ``polya_gamma_mean``),
    c(xi) = log sigmoid(xi) - xi / 2 + omega(xi) xi^2 / 2   (``bound_intercept``),

with equality exactly at xi_i = |eta_i|. For a q over (alpha, beta), the trait's objective is

    L(q, xi) = sum_i [kappa_i E_q eta_i - omega_i E_q eta_i^2 / 2 + c(xi_i)] - KL(q(beta) || p(beta)) + H(q(alpha | beta)),

a lower bound on the Bernoulli evidence (the flat prior on alpha has density 1, so its term is the entropy alone). It
is the only objective this route reports: never the Gaussian ELBO with a noise parameter.

**Two exact coordinate steps.**
1. At fixed q, L is maximized by xi_i^2 = E_q[eta_i^2] (d/dxi of -omega(xi) E eta^2 / 2 + c(xi) is
   -omega'(xi) (E eta^2 - xi^2) / 2 and omega' < 0), and the gain of that update is exact (``BernoulliSites.updated``).
2. At fixed xi, completing the square with the working response z_i = kappa_i / omega_i,

       kappa eta - omega eta^2 / 2 = -omega (z - eta)^2 / 2 + kappa^2 / (2 omega),

   makes L a weighted Gaussian regression with weights W = diag(omega), response z and KNOWN unit noise, plus the
   constant sum_i [kappa_i^2 / (2 omega_i) + c(xi_i)] (``BernoulliSites.constant``). In whitened coordinates
   y~ = W^1/2 z = kappa / sqrt(omega), X~ = W^1/2 X, C~ = W^1/2 C it is the Gaussian model y~ ~ N(C~ alpha + X~ beta, I),
   so every Gaussian solver of this package serves it unchanged, with the noise held at 1, never re-estimated.
Alternating the two is coordinate ascent on L, so L never falls (``bernoulli_ascent`` checks it at every step).
No step here takes the curvature of an integrated likelihood: the bound is exactly quadratic in eta with curvature
omega, and every expectation over q is taken of that quadratic. Where a factor is integrated instead, as
log E[p(y | eta + U)] over a score law U, its curvature in eta is E[d2 l] + Var(d l / d eta), the missing information
included, and the latter term must not be dropped.

**The covariates.** Given beta the exact conditional q(alpha | beta) = N((C'WC)^+ C'W (z - X beta), (C'WC)^+) is
what profiling alpha by the weighted projector H = W^1/2 C (C'WC)^+ C' W^1/2 does. With it,

    E_alpha[-||W^1/2 (z - C alpha - X beta)||^2 / 2] = -||(I - H) W^1/2 (z - X beta)||^2 / 2 - k / 2,
    H(q(alpha | beta)) = k/2 log(2 pi e) - 1/2 log pdet(C'WC),

so the covariates add k/2 log(2 pi) - 1/2 log pdet(C'WC) (``covariate_evidence``, k the rank of W^1/2 C by the
package's rank rule) to the projected Gaussian value. The covariate projection is therefore weight-aware: it is W's
projector, rebuilt whenever xi moves, and anything cached from the projected design (column squares, Grams) belongs
to one set of weights and is keyed by them (``BernoulliSites.key``). Under the joint q,

    E eta_i = z_i - r_i / sqrt(omega_i),   r = (I - H) W^1/2 (z - X E beta),
    Var eta_i = ([Xt Cov(beta) Xt']_ii + H_ii) / omega_i,   Xt = (I - H) W^1/2 X,

(W^1/2 (I - C (C'WC)^+ C'W) = (I - H) W^1/2), which is all the xi update needs from a solver.

**The interface a Gaussian solver implements** (``WeightedGaussianStep``): given the weights and working response it
returns q's per-sample predictor moments and its Gaussian value
E_q[-||W^1/2 (z - eta)||^2 / 2] - KL(q(beta) || p(beta)) + H(q(alpha | beta)). A solver that profiles the covariates
by the projector reports -(||r||^2 + tr(Xt Cov(beta) Xt')) / 2 - KL + ``covariate_evidence(weights, covariates)``. The
mean-field routes (``mean_field.MeanFieldFixedPoints``, ``full_data_fit._FullDataMeanField``) run the same two steps
inside their fixed points, with the xi update in the place of the noise update; ``bernoulli_ascent`` is the driver for
any solver that exposes only this interface (``GaussianPriorStep``, ``CovariateStep``).

**Separation.** With alpha flat the bound has no maximum when the covariates alone (quasi-)separate the labels: xi and
alpha drift without bound. ``logistic_ep.assert_no_separation`` detects it exactly before any fit.

**Calibration and deployment prevalence.** Case-control sampling whose selection depends on y only (not on genotype or
covariates) keeps every logistic slope and moves only the intercept, by log(s_1 / s_0) for the sampling fractions s_y
(Anderson 1972, Biometrika 59:19; Prentice and Pyke 1979, Biometrika 66:403). With sample prevalence p and population
prevalence K that is logit(p) - logit(K), so the deployment model is the fitted one with the logit offset
``ascertainment_offset(p, K)`` = logit(K) - logit(p) added to its intercept. What it assumes: selection into the
training rows depends on the label alone, and K is the deployment population's prevalence of the same outcome over the
same horizon the labels were ascertained on (a label "diagnosed by age a", or within a follow-up window, has the
prevalence of that event, not a lifetime one); the offset is then exact whatever the population's genotype and
covariate distribution, because only the sampling fractions s_1 / s_0 enter. The probability output is the posterior
predictive E_q[sigmoid(eta + shift)] under q's score law, never sigmoid of the posterior mean score
(``fast_scoring.posterior_predictive_probability`` for a Gaussian score law with the draws' and the covariates'
variance; ``fourier_predictive_probability`` for any law given by its characteristic function, such as a mixture's
components), whose shift first matches the
training prevalence (``fast_scoring.predictive_intercept_shift``, the variational predictive's calibration in the large)
and then carries the ascertainment offset. Predictions are evaluated by log loss, Brier score, calibration (the
logistic recalibration's intercept and slope, and calibration in the large) and AUC (``log_loss``, ``brier_score``,
``calibration``, ``auc``).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from scipy.special import expit, log_expit, logit

from sv_pgs._typing import BoolArray, F64Array
from sv_pgs.logistic_ep import _LOG_SIGMOID_CURVATURE_BOUND, assert_no_separation

_EPSILON = float(np.finfo(np.float64).eps)
POLYA_GAMMA_MEAN_AT_ZERO = _LOG_SIGMOID_CURVATURE_BOUND
"""E[PG(1, 0)] = lim tanh(xi / 2) / (2 xi) = 1/4: the logistic curvature at 0, sigmoid(0) sigmoid(-0)."""


# ------------------------------------------------------------------ the Polya-Gamma bound


def polya_gamma_mean(xi: F64Array) -> F64Array:
    """E[omega] for omega ~ PG(1, xi): tanh(xi / 2) / (2 xi), and its limit 1/4 at xi = 0.

    tanh(xi / 2) = -expm1(-xi) / (1 + e^-xi) for xi >= 0 keeps every digit as xi -> 0, where tanh(xi / 2) / xi would
    lose them to the cancellation in 1 - e^-xi; the bound depends on |xi| only."""
    magnitude = np.abs(np.asarray(xi, dtype=np.float64))
    positive = magnitude > 0.0
    safe = np.where(positive, magnitude, 1.0)
    return np.where(positive, -np.expm1(-safe) / ((1.0 + np.exp(-safe)) * 2.0 * safe), POLYA_GAMMA_MEAN_AT_ZERO)


def bound_intercept(xi: F64Array) -> F64Array:
    """c(xi) = log sigmoid(xi) - xi / 2 + omega(xi) xi^2 / 2, the bound's value at eta = 0 (module docstring).

    log sigmoid(|xi|) - |xi| / 2 = -(|xi| / 2 + log1p(e^-|xi|)), which never overflows."""
    magnitude = np.abs(np.asarray(xi, dtype=np.float64))
    return -(0.5 * magnitude + np.log1p(np.exp(-magnitude))) + 0.5 * polya_gamma_mean(magnitude) * magnitude * magnitude


def bernoulli_log_likelihood(labels: F64Array, predictor: F64Array) -> F64Array:
    """log p(y_i | eta_i) = y log sigmoid(eta) + (1 - y) log sigmoid(-eta), exactly (``log_expit``)."""
    outcome = np.asarray(labels, dtype=np.float64)
    eta = np.asarray(predictor, dtype=np.float64)
    return np.where(outcome > 0.5, log_expit(eta), log_expit(-eta))


def site_bound(labels: F64Array, xi: F64Array, mean: F64Array, second_moment: F64Array) -> F64Array:
    """Each sample's term of L at q: kappa E eta - omega E eta^2 / 2 + c(xi), a lower bound on E_q log p(y | eta)."""
    kappa = np.asarray(labels, dtype=np.float64) - 0.5
    return kappa * mean - 0.5 * polya_gamma_mean(xi) * second_moment + bound_intercept(xi)


def covariate_evidence(weights: F64Array, covariates: F64Array) -> float:
    """The covariates' term of L: k/2 log(2 pi) - 1/2 log pdet(C'WC) (module docstring), on the rows with weight.

    pdet over the numerical column space of W^1/2 C, by the rank rule of ``dual_solve.covariate_whitener`` (singular
    values above max(n, k) eps s_max, Golub and Van Loan 5.4.1), from the R factor of W^1/2 C = QR (backward stable,
    and the Gram's squared condition number never forms)."""
    weight = np.asarray(weights, dtype=np.float64)
    rows = weight > 0.0
    design = np.sqrt(weight[rows])[:, None] * np.asarray(covariates, dtype=np.float64)[rows]
    if design.shape[1] == 0:
        return 0.0
    triangle = np.linalg.qr(design, mode="r")
    singular = np.linalg.svd(triangle, compute_uv=False)
    kept = singular > max(design.shape) * _EPSILON * singular[0]
    return 0.5 * int(np.count_nonzero(kept)) * float(np.log(2.0 * np.pi)) - float(np.sum(np.log(singular[kept])))


@dataclass(frozen=True)
class BernoulliSites:
    """The variational Polya-Gamma factors q(omega_i) = PG(1, xi_i) of one binary model over the samples.

    ``labels`` (n,) are 0/1 on the ``training`` rows (read nowhere else), ``xi`` (n,) >= 0. Off the training rows the
    weight and the working response are 0, so a weighted metric over all samples is the model's own."""

    labels: F64Array
    training: BoolArray
    xi: F64Array
    key: bytes = field(init=False, compare=False)

    def __post_init__(self) -> None:
        training = np.asarray(self.training, dtype=bool)
        labels = np.where(training, np.asarray(self.labels, dtype=np.float64), 0.0)
        if not np.all((labels == 0.0) | (labels == 1.0)):
            raise ValueError("a binary model's training labels must be 0 or 1.")
        xi = np.where(training, np.abs(np.asarray(self.xi, dtype=np.float64)), 0.0)
        if not np.all(np.isfinite(xi)):
            raise ValueError("the Polya-Gamma parameters must be finite.")
        object.__setattr__(self, "labels", labels)
        object.__setattr__(self, "training", training)
        object.__setattr__(self, "xi", xi)
        # The weights are a function of xi on the mask, so this names the weighted metric (its projector, column
        # squares and Grams) for every cache that holds one.
        object.__setattr__(self, "key", hashlib.sha256(training.tobytes() + xi.tobytes()).digest())

    @classmethod
    def start(cls, labels: F64Array, training: BoolArray | None = None) -> "BernoulliSites":
        """xi = 0: every weight 1/4, the equal-weight Gaussian problem with response 4 kappa = 4 y - 2 (noise 4 in y's
        units), which is where Stage 0's statistics are exact."""
        values = np.asarray(labels, dtype=np.float64)
        mask = np.ones(values.shape[0], dtype=bool) if training is None else np.asarray(training, dtype=bool)
        return cls(labels=values, training=mask, xi=np.zeros(values.shape[0]))

    @property
    def kappa(self) -> F64Array:
        return np.where(self.training, self.labels - 0.5, 0.0)

    @property
    def weights(self) -> F64Array:
        """omega_i on the training rows, 0 elsewhere."""
        return np.where(self.training, polya_gamma_mean(self.xi), 0.0)

    @property
    def response(self) -> F64Array:
        """z_i = kappa_i / omega_i on the training rows, 0 elsewhere."""
        weights = self.weights
        return np.where(self.training, self.kappa / np.where(self.training, weights, 1.0), 0.0)

    @property
    def root_response(self) -> F64Array:
        """y~_i = kappa_i / sqrt(omega_i), the whitened working response W^1/2 z."""
        weights = self.weights
        return np.where(self.training, self.kappa / np.sqrt(np.where(self.training, weights, 1.0)), 0.0)

    def constant(self) -> float:
        """sum_i [kappa_i^2 / (2 omega_i) + c(xi_i)] over the training rows: L less the weighted Gaussian value."""
        weights = self.weights[self.training]
        kappa = self.kappa[self.training]
        return float(np.sum(kappa * kappa / (2.0 * weights) + bound_intercept(self.xi[self.training])))

    def constant_size(self) -> float:
        """The sum of the constant's terms' magnitudes, for the objective's rounding bound."""
        weights = self.weights[self.training]
        kappa = self.kappa[self.training]
        return float(np.sum(kappa * kappa / (2.0 * weights) + np.abs(bound_intercept(self.xi[self.training]))))

    def value(self, mean: F64Array, variance: F64Array) -> float:
        """sum_i over the training rows of ``site_bound`` at q's predictor moments."""
        rows = self.training
        return float(np.sum(site_bound(self.labels[rows], self.xi[rows], mean[rows], mean[rows] ** 2 + variance[rows])))

    def updated(self, mean: F64Array, variance: F64Array) -> tuple["BernoulliSites", float]:
        """(the sites at xi_i^2 = E eta_i^2 for q's predictor moments, the exact gain of L from the update at fixed q).

        The gain is sum_i [site_bound(xi') - site_bound(xi)] at the same moments, >= 0 because xi' maximizes each term;
        it is formed term by term, so it never cancels between two large totals."""
        moments = np.asarray(mean, dtype=np.float64)
        spread = np.asarray(variance, dtype=np.float64)
        second = moments * moments + spread
        xi = np.where(self.training, np.sqrt(np.where(self.training, second, 0.0)), 0.0)
        rows = self.training
        gain = float(np.sum(
            site_bound(self.labels[rows], xi[rows], moments[rows], second[rows])
            - site_bound(self.labels[rows], self.xi[rows], moments[rows], second[rows])
        ))
        return BernoulliSites(labels=self.labels, training=self.training, xi=xi), max(gain, 0.0)


# ------------------------------------------------------------------ the weighted Gaussian step


@dataclass(frozen=True)
class WeightedGaussianState:
    """What a weighted Gaussian step returns (module docstring): q's predictor moments E eta_i and Var eta_i (n,) under
    the joint q(alpha, beta), and ``value`` = E_q[-||W^1/2 (z - eta)||^2 / 2] - KL(q(beta) || p(beta)) + H(q(alpha | beta)),
    with ``rounding`` a bound on its rounding error. ``detail`` is the solver's own state, returned untouched."""

    predictor_mean: F64Array
    predictor_variance: F64Array
    value: float
    rounding: float = 0.0
    detail: object = None


class WeightedGaussianStep(Protocol):
    """A Gaussian solver of the whitened problem at fixed Polya-Gamma weights (module docstring).

    ``step(weights, response)`` maximizes (or improves) the Gaussian value over q at the weights omega (n,), zero off
    the training rows, and the working response z = kappa / omega, with the noise variance fixed at 1, and returns its
    ``WeightedGaussianState``. Its covariate projection must be W's, and anything it caches from the projected design
    must be keyed by the weights. A Gaussian route implements it by running its Gaussian updates on X~ = W^1/2 X and
    y~ = W^1/2 z at unit noise."""

    def __call__(self, weights: F64Array, response: F64Array) -> WeightedGaussianState:
        ...


@dataclass(frozen=True)
class BernoulliAscent:
    """``bernoulli_ascent``'s result: the final sites and state, L there, and L after every step (``history``: each
    Gaussian step then each xi update, non-decreasing), with the last xi update's gain (the remaining gain)."""

    sites: BernoulliSites
    state: WeightedGaussianState
    bound: float
    history: tuple[float, ...]
    remaining_gain: float
    steps: int


class BoundDecreased(FloatingPointError):
    """A step lowered the Bernoulli bound beyond its rounding: the ascent is broken."""


def bernoulli_ascent(step: WeightedGaussianStep, sites: BernoulliSites, tolerance: float) -> BernoulliAscent:
    """Coordinate ascent on L (module docstring): the Gaussian step at the current weights, then the xi update, until the
    xi update's exact gain is at most ``tolerance`` nats (the fit's resolution, 1/(2K) for K posterior draws).

    Each Gaussian step starts where the last xi update left L, L(q, xi'), and must not end below it, so the recorded
    bound is monotone to the steps' rounding; a fall beyond it raises ``BoundDecreased``. Where the step is an exact
    maximizer at fixed xi (``GaussianPriorStep``) the stopping gain is the whole remaining gain of the coordinate
    ascent's last cycle; a step that only improves hands its own remainder to its caller through ``detail``."""
    history: list[float] = []
    count = 0
    reached: float | None = None
    while True:
        state = step(sites.weights, sites.response)
        count += 1
        bound = state.value + sites.constant()
        rounding = state.rounding + (sites.training.sum() + 1) * _EPSILON * sites.constant_size()
        if reached is not None and bound < reached - rounding:
            raise BoundDecreased(f"a weighted Gaussian step lowered the Bernoulli bound by {reached - bound:.3g} nats")
        history.append(bound)
        updated, gain = sites.updated(state.predictor_mean, state.predictor_variance)
        if gain <= tolerance:
            return BernoulliAscent(sites=sites, state=state, bound=bound, history=tuple(history), remaining_gain=gain, steps=count)
        sites = updated
        reached = bound + gain
        history.append(reached)


def _whitened(weights: F64Array, response: F64Array, covariates: F64Array) -> tuple[BoolArray, F64Array, F64Array, F64Array]:
    """(the rows with weight, sqrt(omega) there, an orthonormal basis Q of W^1/2 C there, y~_P = (I - QQ') W^1/2 z)."""
    rows = np.asarray(weights) > 0.0
    root = np.sqrt(np.asarray(weights, dtype=np.float64)[rows])
    design = root[:, None] * np.asarray(covariates, dtype=np.float64)[rows]
    basis = np.zeros((design.shape[0], 0))
    if design.shape[1]:
        left, singular, _right = np.linalg.svd(design, full_matrices=False)
        basis = left[:, singular > max(design.shape) * _EPSILON * singular[0]]
    whitened = root * np.asarray(response, dtype=np.float64)[rows]
    return rows, root, basis, whitened - basis @ (basis.T @ whitened)


class CovariateStep:
    """The covariates alone (no genetic effect): q(alpha) = N((C'WC)^+ C'W z, (C'WC)^+), exact at each xi. Its value
    is -||(I - H) W^1/2 z||^2 / 2 + ``covariate_evidence``; E eta = z - r / sqrt(omega), Var eta = H_ii / omega. The
    null genetic model of a binary trait is ``bernoulli_ascent`` over this step."""

    def __init__(self, covariates: F64Array) -> None:
        self.covariates = np.asarray(covariates, dtype=np.float64)

    def __call__(self, weights: F64Array, response: F64Array) -> WeightedGaussianState:
        rows, root, basis, residual = _whitened(weights, response, self.covariates)
        mean = np.zeros(rows.shape[0])
        variance = np.zeros(rows.shape[0])
        mean[rows] = np.asarray(response, dtype=np.float64)[rows] - residual / root
        variance[rows] = np.sum(basis * basis, axis=1) / (root * root)
        square = float(residual @ residual)
        value = -0.5 * square + covariate_evidence(weights, self.covariates)
        return WeightedGaussianState(mean, variance, value, rounding=(rows.sum() + 1) * _EPSILON * square)

    def coefficients(self, weights: F64Array, response: F64Array) -> tuple[F64Array, F64Array]:
        """(q(alpha)'s mean (C'WC)^+ C'W z and covariance (C'WC)^+), by the pseudo-inverse's rank rule."""
        rows = np.asarray(weights) > 0.0
        weight = np.asarray(weights, dtype=np.float64)[rows]
        design = self.covariates[rows]
        gram = design.T @ (weight[:, None] * design)
        inverse = np.linalg.pinv(gram, rcond=max(design.shape) * _EPSILON, hermitian=True)
        return inverse @ (design.T @ (weight * np.asarray(response, dtype=np.float64)[rows])), inverse


class GaussianPriorStep:
    """The prior family's Gaussian member at fixed weights, exactly (the Jaakkola-Jordan variational logistic
    regression with a Gaussian prior): beta_j ~ N(0, t u_j) on the dense ``genotypes`` (n x p, standardized columns),
    u_j = ``relative_variances``, and t maximizing the step's own value (its marginal likelihood at unit noise), so the
    step is an exact maximizer of L over (q, t) at fixed xi. With Xt = (I - H) W^1/2 X, K = Xt U Xt' = V diag(lambda) V':

        value = -1/2 sum_k log(1 + t lambda_k) - 1/2 y~_P'(I + t K)^-1 y~_P + covariate_evidence,
        Xt Cov(beta) Xt' = t K (I + t K)^-1,   E beta = t U Xt'(I + t K)^-1 y~_P,

    (the Gaussian evidence of y~_P over the covariates' complement, q the exact conditional posterior). t is searched
    over the resolvable range of K's nonzero spectrum, [sqrt(eps) / lambda_max, 1 / (sqrt(eps) lambda_min)], as
    ``small_n.GaussianMember`` does. The dense reference of the binary route: tests and the dense end of a mixture."""

    def __init__(self, genotypes: F64Array, covariates: F64Array, relative_variances: F64Array) -> None:
        self.genotypes = np.asarray(genotypes, dtype=np.float64)
        self.covariates = np.asarray(covariates, dtype=np.float64)
        self.relative_variances = np.asarray(relative_variances, dtype=np.float64)

    def __call__(self, weights: F64Array, response: F64Array) -> WeightedGaussianState:
        from scipy.optimize import minimize_scalar

        rows, root, basis, target = _whitened(weights, response, self.covariates)
        design = root[:, None] * self.genotypes[rows]
        design = design - basis @ (basis.T @ design)
        values, vectors = np.linalg.eigh((design * self.relative_variances[None, :]) @ design.T)
        resolvable = values > values[-1] * design.shape[0] * _EPSILON
        values, vectors = values[resolvable], vectors[:, resolvable]
        rotated = vectors.T @ target
        remainder = float(target @ target - rotated @ rotated)

        def negative(log_t: float) -> float:
            scale = float(np.exp(log_t))
            return 0.5 * (float(np.sum(np.log1p(scale * values))) + float(np.sum(rotated ** 2 / (1.0 + scale * values))) + remainder)

        half = np.sqrt(_EPSILON)
        bounds = (float(np.log(half / values[-1])), float(np.log(1.0 / (half * values[0]))))
        scale = float(np.exp(minimize_scalar(negative, bounds=bounds, method="bounded").x))
        shrunk = rotated / (1.0 + scale * values)
        residual = vectors @ shrunk + (target - vectors @ rotated)
        effects = scale * self.relative_variances * (design.T @ (vectors @ shrunk))
        leverage = np.sum(basis * basis, axis=1)
        explained = np.sum(vectors * vectors * (scale * values / (1.0 + scale * values))[None, :], axis=1)
        mean = np.zeros(rows.shape[0])
        variance = np.zeros(rows.shape[0])
        mean[rows] = np.asarray(response, dtype=np.float64)[rows] - residual / root
        variance[rows] = (explained + leverage) / (root * root)
        value = -negative(float(np.log(scale))) + covariate_evidence(weights, self.covariates)
        size = float(np.sum(np.log1p(scale * values))) + float(target @ target)
        detail = {"scale": scale, "effects": effects, "values": values, "vectors": vectors, "design": design}
        return WeightedGaussianState(mean, variance, value, rounding=(rows.sum() + values.shape[0]) * _EPSILON * size, detail=detail)


def fit_binary_covariates(covariates: F64Array, labels: F64Array, tolerance: float) -> tuple[BernoulliAscent, F64Array, F64Array]:
    """A binary model with no genetic column (the null genetic model): the separation check, then ``bernoulli_ascent``
    over ``CovariateStep``. Returns (the ascent, q(alpha)'s mean and covariance at its final sites)."""
    covariate_matrix = np.asarray(covariates, dtype=np.float64)
    outcome = np.asarray(labels, dtype=np.float64)
    assert_no_separation(covariate_matrix, outcome)
    step = CovariateStep(covariate_matrix)
    ascent = bernoulli_ascent(step, BernoulliSites.start(outcome), tolerance)
    alpha, covariance = step.coefficients(ascent.sites.weights, ascent.sites.response)
    return ascent, alpha, covariance


# ------------------------------------------------------------------ calibration


def ascertainment_offset(sample_prevalence: float, population_prevalence: float) -> float:
    """logit(K) - logit(p): the intercept's move from a case-control sample of prevalence p to a population of
    prevalence K (module docstring: Anderson 1972; Prentice and Pyke 1979). 0 when the sample is the population's."""
    sample, population = float(sample_prevalence), float(population_prevalence)
    if not (0.0 < sample < 1.0 and 0.0 < population < 1.0):
        raise ValueError("prevalences must lie strictly between 0 and 1.")
    return float(logit(population) - logit(sample))


def training_prevalence(labels: F64Array) -> float:
    outcome = np.asarray(labels, dtype=np.float64)
    prevalence = float(np.mean(outcome))
    if not 0.0 < prevalence < 1.0:
        raise ValueError("a binary model's training labels need both cases and controls.")
    return prevalence


def probability(linear_predictor: F64Array, predictor_variance: F64Array, intercept_shift: float) -> F64Array:
    """P(y = 1) = E[sigmoid(eta + shift)] under eta ~ N(linear predictor, variance): the posterior predictive
    (``fast_scoring.posterior_predictive_probability``, to 2 eps by its a-priori trapezoid rule)."""
    from sv_pgs.fast_scoring import posterior_predictive_probability

    return posterior_predictive_probability(linear_predictor, predictor_variance, intercept_shift)


FOURIER_FREQUENCY_LIMIT = float(np.log(2.0 / (np.pi * _EPSILON)) / np.pi)
"""T with the omitted tail of the Fourier form at most eps: log1p(2 / expm1(pi T)) / pi <= 2 e^(-pi T) / pi = eps."""


def fourier_predictive_probability(
    linear_predictor: F64Array, component_weights: F64Array, component_means: F64Array, component_variances: F64Array,
) -> tuple[F64Array, F64Array]:
    """The posterior predictive P(y = 1) = E[sigmoid(eta + U)] for a score law U given by its characteristic function,
    here a Gaussian mixture phi_U(t) = sum_c w_c exp(i t mu_c - t^2 s_c^2 / 2) per sample (a mixture fit's components;
    one component is the Gaussian law of ``fast_scoring.posterior_predictive_probability``), by

        E[sigmoid(eta + U)] = 1/2 + int_0^inf Im(e^(i t eta) phi_U(t)) / sinh(pi t) dt,

    (sigmoid(x) - 1/2 = (1/2) tanh(x / 2), whose Fourier sine transform is pi / sinh(pi t)). |Im(...)| <= 1, so the
    frequencies beyond T omit at most int_T^inf dt / sinh(pi t) = log1p(2 / expm1(pi T)) / pi, whatever the number of
    effects summed into U; T is ``FOURIER_FREQUENCY_LIMIT``, where that is eps. On [0, T] the integrand is analytic
    (sinh's nearest zero off the real line is at t = i) and finite at 0, so Gauss-Legendre converges geometrically: the
    node count doubles until two successive rules agree to eps (the returned bound adds the tail's).

    ``linear_predictor`` is (n,), the component arrays (n, C) or (C,) with weights summing to 1. Returns (P(y = 1),
    a bound on its error)."""
    eta = np.asarray(linear_predictor, dtype=np.float64)
    weights = np.broadcast_to(np.asarray(component_weights, dtype=np.float64), eta.shape + np.shape(component_weights)[-1:])
    means = np.broadcast_to(np.asarray(component_means, dtype=np.float64), weights.shape)
    variances = np.broadcast_to(np.asarray(component_variances, dtype=np.float64), weights.shape)
    tail = float(np.log1p(2.0 / np.expm1(np.pi * FOURIER_FREQUENCY_LIMIT)) / np.pi)

    def rule(count: int) -> F64Array:
        nodes, node_weights = np.polynomial.legendre.leggauss(count)
        t = 0.5 * FOURIER_FREQUENCY_LIMIT * (nodes + 1.0)
        scaled = 0.5 * FOURIER_FREQUENCY_LIMIT * node_weights / np.sinh(np.pi * t)
        # Im(e^{i t eta} sum_c w_c e^{i t mu_c - t^2 s_c^2 / 2}) = sum_c w_c e^{-t^2 s_c^2 / 2} sin(t (eta + mu_c)).
        phase = t[None, None, :] * (eta[:, None, None] + means[:, :, None])
        damping = np.exp(-0.5 * t[None, None, :] ** 2 * variances[:, :, None])
        values = np.sum(weights[:, :, None] * damping * np.sin(phase), axis=1)
        return 0.5 + values @ scaled

    count = 2
    previous = rule(count)
    while True:
        count *= 2
        current = rule(count)
        change = float(np.max(np.abs(current - previous)))
        if change <= _EPSILON * count:
            return current, np.full(eta.shape, change + tail)
        previous = current


def calibrated_shift(
    predictor_mean: F64Array, predictor_variance: F64Array, labels: F64Array, population_prevalence: float | None = None,
) -> float:
    """The predictive's intercept shift: matched to the training prevalence on the training rows' predictive moments
    (``fast_scoring.predictive_intercept_shift``), plus the ascertainment offset to ``population_prevalence`` where the
    training sample is a case-control sample of a population with that prevalence (None: the sample is the population)."""
    from sv_pgs.fast_scoring import predictive_intercept_shift

    shift = predictive_intercept_shift(predictor_mean, predictor_variance, labels)
    if population_prevalence is None:
        return shift
    return shift + ascertainment_offset(training_prevalence(labels), population_prevalence)


# ------------------------------------------------------------------ evaluation


def log_loss(labels: F64Array, probabilities: F64Array) -> float:
    """Mean -log p(y) of the predicted probabilities (nats per sample); a probability of exactly 0 or 1 on the wrong
    label is an infinite loss, reported as such."""
    outcome = np.asarray(labels, dtype=np.float64)
    chance = np.asarray(probabilities, dtype=np.float64)
    with np.errstate(divide="ignore"):
        return float(-np.mean(np.where(outcome > 0.5, np.log(chance), np.log1p(-chance))))


def brier_score(labels: F64Array, probabilities: F64Array) -> float:
    """Mean (p - y)^2."""
    difference = np.asarray(probabilities, dtype=np.float64) - np.asarray(labels, dtype=np.float64)
    return float(np.mean(difference * difference))


def auc(labels: F64Array, scores: F64Array) -> float:
    """P(score of a case > score of a control) + P(tie) / 2, by the Mann-Whitney rank sum with average ranks."""
    from scipy.stats import rankdata

    outcome = np.asarray(labels, dtype=np.float64) > 0.5
    cases, controls = int(outcome.sum()), int((~outcome).sum())
    if not cases or not controls:
        raise ValueError("the AUC needs both cases and controls.")
    ranks = rankdata(np.asarray(scores, dtype=np.float64), method="average")
    return float((ranks[outcome].sum() - cases * (cases + 1) / 2.0) / (cases * controls))


@dataclass(frozen=True)
class Calibration:
    """The logistic recalibration logit P(y = 1) = a + b logit(p) by maximum likelihood (a perfectly calibrated p has
    a = 0, b = 1; b < 1 means over-dispersed predictions), and calibration in the large, mean(y) - mean(p)."""

    intercept: float
    slope: float
    in_the_large: float


def calibration(labels: F64Array, probabilities: F64Array) -> Calibration:
    """``Calibration`` by Newton's method on the two-parameter logistic likelihood (concave), stopped when the Newton
    decrement falls to the log-likelihood's own rounding, n eps (sum_i |log-likelihood term|). The covariates' exact
    separation test comes first: a single score that separates the labels has no finite recalibration."""
    outcome = np.asarray(labels, dtype=np.float64)
    chance = np.asarray(probabilities, dtype=np.float64)
    score = logit(chance)
    design = np.column_stack([np.ones(outcome.shape[0]), score])
    assert_no_separation(design, outcome)
    coefficients = np.array([0.0, 1.0])
    while True:
        predictor = design @ coefficients
        fitted = expit(predictor)
        gradient = design.T @ (outcome - fitted)
        hessian = design.T @ ((fitted * (1.0 - fitted))[:, None] * design)
        step = np.linalg.solve(hessian, gradient)
        decrement = float(gradient @ step)
        resolution = outcome.shape[0] * _EPSILON * float(np.sum(np.abs(bernoulli_log_likelihood(outcome, predictor))))
        if decrement <= resolution:
            break
        # The log-likelihood is concave, so a halved Newton step ascends wherever the full one overshoots.
        current = float(np.sum(bernoulli_log_likelihood(outcome, predictor)))
        length = 1.0
        while float(np.sum(bernoulli_log_likelihood(outcome, design @ (coefficients + length * step)))) < current and length * decrement > resolution:
            length *= 0.5
        coefficients = coefficients + length * step
    return Calibration(intercept=float(coefficients[0]), slope=float(coefficients[1]), in_the_large=float(outcome.mean() - chance.mean()))
