"""The laws a scoring model's K posterior draws come from, read a tile of rows at a time.

A fitted model carries K draws beta^(k) of its effects (``fast_scoring``: each is one more weight column of the
score). Held as a rows x K matrix they are the scoring route's largest array at biobank scale (518k members x 64 draws
is 265 MB per model, and a mixture of components held one such matrix per component), and nothing needs them all at
once: the scorer reads the store a block of rows at a time and needs only that block's rows of every draw. So a model's
draws are a law that yields any tile of rows on demand:

- ``DenseDraws``: a rows x K matrix already drawn (the full-data EP route's perturb-and-solve draws, which are joint
  over every row and so exist only whole; the small-n routes, whose problems the dense router bounds). A tile is a
  view.
- ``ProductMixtureDraws``: the mean-field route's mixture q = sum_c w_c q_c of products q_c = prod_j q_cj, each
  member's q_cj the prior's scale mixture tilted by its pseudo-likelihood (omega_cj, h_cj). Its parameters are
  O(C p), and a tile's draws are generated when asked.

Sampling law of ``ProductMixtureDraws``. Draw k is one sample of the mixture: its component c_k ~ Categorical(w) (an
honest draw from the weights; the draws' components are independent, so the share of draws from component c is
Binomial(K, w_c), unbiased for w_c at every K), then every member independently from q_(c_k) j: its lattice node from
the node responsibilities at (omega, h) by inverse CDF, then N(h c_node, c_node) with c_node the node's conditional
variance (``scale_mixture_ep._components``; the law of ``mean_field.product_law``). This replaces the largest-remainder
quotas the draws were allotted by before (deterministic counts round(w_c K), which give a component of weight below
1 / (2K) no draw at all and one of weight 1 - 1/(3K) all of them, so the draws' mixture was not q's).

Reproducibility. Every uniform comes from Philox4x32-10 (Salmon, Moraes, Dror and Shaw 2011, "Parallel random numbers:
as easy as 1, 2, 3"), a counter-based generator: the 128-bit output is a fixed function of the key (the model's seed)
and the counter (row, draw, stream), so draw k of row j is the same number whatever tile, tile order, device or thread
count produced it. The component of draw k uses the counter (0, 0, k, COMPONENT_STREAM); member j's node and normal
use the counter (j mod 2^32, j div 2^32, k, MEMBER_STREAM), the first two output words for the node's uniform and the
last two for the normal's (by the inverse normal CDF, so one counter gives one normal).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
from scipy.special import ndtri

from sv_pgs._typing import F64Array, I64Array

_FLOAT64_BYTES = np.dtype(np.float64).itemsize

# Philox4x32-10 (Salmon et al. 2011, Random123's philox.h): the round multipliers, the Weyl key increments and the
# round count; outputs are 32-bit words.
_PHILOX_MULTIPLIERS = (0xD2511F53, 0xCD9E8D57)
_PHILOX_WEYL = (0x9E3779B9, 0xBB67AE85)
_PHILOX_ROUNDS = 10
_WORD_BITS = 32
_WORD_MASK = np.uint64(0xFFFFFFFF)
# A 53-bit uniform from two 32-bit words (genrand_res53, Matsumoto and Nishimura's MT19937 reference): the top 27 bits
# of the first word and the top 26 of the second, (a 2^26 + b) / 2^53, the full float64 mantissa.
_HIGH_SHIFT = 5
_LOW_SHIFT = 6
_LOW_BITS = 26
_MANTISSA_BITS = 53

MEMBER_STREAM = 0
COMPONENT_STREAM = 1


def philox4x32(counter: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray], key: tuple[int, int]) -> tuple[np.ndarray, ...]:
    """Philox4x32-10 of broadcastable counter words (uint64 arrays holding 32-bit values) under a 64-bit key given as
    two 32-bit words: the four 32-bit output words, as uint64 arrays."""
    c0, c1, c2, c3 = (np.asarray(word, dtype=np.uint64) & _WORD_MASK for word in counter)
    k0, k1 = np.uint64(key[0]) & _WORD_MASK, np.uint64(key[1]) & _WORD_MASK
    first, second = (np.uint64(multiplier) for multiplier in _PHILOX_MULTIPLIERS)
    shift = np.uint64(_WORD_BITS)
    for round_index in range(_PHILOX_ROUNDS):
        if round_index:
            k0 = (k0 + np.uint64(_PHILOX_WEYL[0])) & _WORD_MASK
            k1 = (k1 + np.uint64(_PHILOX_WEYL[1])) & _WORD_MASK
        product0 = first * c0
        product1 = second * c2
        c0, c1, c2, c3 = (
            (product1 >> shift) ^ c1 ^ k0,
            product1 & _WORD_MASK,
            (product0 >> shift) ^ c3 ^ k1,
            product0 & _WORD_MASK,
        )
    return c0, c1, c2, c3


def _uniform(high: np.ndarray, low: np.ndarray, open_below: bool) -> np.ndarray:
    """The 53-bit uniform of two words on [0, 1), or on (0, 1) (shifted by half its spacing) where ``open_below``."""
    mantissa = (high >> np.uint64(_HIGH_SHIFT)) * np.uint64(1 << _LOW_BITS) + (low >> np.uint64(_LOW_SHIFT))
    value = mantissa.astype(np.float64)
    if open_below:
        value += 0.5
    return value / float(1 << _MANTISSA_BITS)


def seed_key(*keys: int) -> tuple[int, int]:
    """A Philox key (two 32-bit words) from integers, by ``SeedSequence``."""
    words = np.random.SeedSequence([int(key) for key in keys]).generate_state(2, dtype=np.uint32)
    return int(words[0]), int(words[1])


class DrawLaw(Protocol):
    """K draws of a model's effects over its rows, read by tiles of rows."""

    kind: str

    @property
    def shape(self) -> tuple[int, int]: ...

    def tile(self, start: int, stop: int) -> F64Array:
        """Rows start..stop-1 of every draw, (stop - start) x K."""

    def tile_row_bytes(self) -> int:
        """Working bytes per row of a tile beyond the tile itself."""

    def arrays(self) -> dict[str, np.ndarray]:
        """The law's parameters, by name (``artifact``)."""


@dataclass(frozen=True)
class DenseDraws:
    """Draws already drawn, rows x K."""

    values: F64Array
    kind: str = "dense"

    def __post_init__(self) -> None:
        values = np.asarray(self.values)
        if values.ndim != 2 or values.dtype != np.float64:
            raise ValueError("dense draws must be a float64 [rows, draws] matrix.")
        if not np.all(np.isfinite(values)):
            raise ValueError("posterior draws must be finite.")

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.values.shape[0]), int(self.values.shape[1]))

    def tile(self, start: int, stop: int) -> F64Array:
        return self.values[start:stop]

    def tile_row_bytes(self) -> int:
        return 0

    def subset(self, rows: np.ndarray) -> "DenseDraws":
        """The draws of ``rows`` (indices or a mask)."""
        return DenseDraws(np.ascontiguousarray(self.values[rows]))

    def arrays(self) -> dict[str, np.ndarray]:
        return {"values": self.values}

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        return np.asarray(self.values, dtype=dtype)


# A tile row's draw-wide arrays live at once in ``ProductMixtureDraws.tile`` (8-byte values each): Philox's four
# counter words and two products, the node uniforms, the normals, the nodes, their conditional variances and the
# component's values before they are placed.
_DRAW_INTERMEDIATES = ("word0", "word1", "word2", "word3", "product0", "product1", "node_uniform", "normal", "node", "conditional", "values")


@dataclass(frozen=True)
class ProductMixtureDraws:
    """The mean-field mixture's law (module docstring). Per member j: ``class_index`` and ``log_scale`` log u_j; per
    class the lattice's ``log_density`` (classes x nodes) on ``log_variance_grid``; per component c: ``shift`` h_c and
    ``omega`` omega_c = ||x_j||^2 / sigma_c^2 in the component's own metric (components x members: a binary model's
    components each have their own weights) and ``weights`` w_c; ``key`` the Philox key and ``draw_count`` K. Row j of
    the law is member j."""

    class_index: I64Array
    log_scale: F64Array
    log_density: F64Array
    log_variance_grid: F64Array
    shift: F64Array
    omega: F64Array
    weights: F64Array
    key: tuple[int, int]
    draw_count: int
    # Each row's member number for its counters (None: row j is member j), so a subset of the rows draws the very
    # numbers the whole law draws for them (``subset``).
    row_ids: I64Array | None = None
    kind: str = "product_mixture"

    def __post_init__(self) -> None:
        members = int(np.asarray(self.class_index).shape[0])
        components = int(np.asarray(self.weights).shape[0])
        if np.shape(self.log_scale) != (members,):
            raise ValueError("log_scale needs one entry per member.")
        if np.shape(self.shift) != (components, members) or np.shape(self.omega) != (components, members):
            raise ValueError("shift and omega must be components x members.")
        weights = np.asarray(self.weights, dtype=np.float64)
        if components == 0 or np.any(weights < 0.0) or not np.isclose(weights.sum(), 1.0, rtol=components * np.finfo(np.float64).eps, atol=0.0):
            raise ValueError("the component weights must be a probability vector.")
        density = np.asarray(self.log_density)
        if density.ndim != 2 or density.shape[1] != np.shape(self.log_variance_grid)[0] or int(np.max(self.class_index, initial=-1)) >= density.shape[0]:
            raise ValueError("log_density must be classes x nodes on log_variance_grid, with a row for every member's class.")
        if not (np.all(np.isfinite(self.shift)) and np.all(np.isfinite(self.omega)) and np.all(np.asarray(self.omega) >= 0.0)):
            raise ValueError("component shifts must be finite and precisions non-negative.")
        if int(self.draw_count) < 0:
            raise ValueError("draw_count must be non-negative.")

    @property
    def shape(self) -> tuple[int, int]:
        return (int(np.asarray(self.class_index).shape[0]), int(self.draw_count))

    def components_of_draws(self) -> I64Array:
        """c_k ~ Categorical(w) for each draw k, by inverse CDF of its counter's uniform."""
        draws = np.arange(int(self.draw_count), dtype=np.uint64)
        zero = np.zeros_like(draws)
        words = philox4x32((zero, zero, draws, np.full_like(draws, COMPONENT_STREAM)), self.key)
        uniform = _uniform(words[0], words[1], open_below=False)
        cumulative = np.cumsum(np.asarray(self.weights, dtype=np.float64))
        return np.minimum(np.searchsorted(cumulative, uniform, side="right"), cumulative.shape[0] - 1).astype(np.int64)

    def tile_row_bytes(self) -> int:
        """A tile row's working bytes: the kernel rows' node-wide intermediates (``scale_mixture_ep._ROW_INTERMEDIATES``
        over the lattice) and the draw-wide ones (``_DRAW_INTERMEDIATES`` over K)."""
        from sv_pgs.scale_mixture_ep import _ROW_INTERMEDIATES

        grid = int(np.shape(self.log_variance_grid)[0])
        return _FLOAT64_BYTES * (_ROW_INTERMEDIATES * grid + len(_DRAW_INTERMEDIATES) * int(self.draw_count))

    def tile(self, start: int, stop: int) -> F64Array:
        """Rows start..stop-1 of every draw: per component present among the draws and per class present in the rows,
        the kernel rows' responsibilities and conditional variances, then each (row, draw)'s node and normal from its
        own counter."""
        from sv_pgs.mean_field import _sample_nodes
        from sv_pgs.scale_mixture_ep import _components

        rows = np.arange(int(start), int(stop), dtype=np.int64)
        count = int(self.draw_count)
        out = np.empty((rows.shape[0], count))
        if not rows.size or not count:
            return out
        components = self.components_of_draws()
        ids = (rows if self.row_ids is None else np.asarray(self.row_ids, dtype=np.int64)[rows]).astype(np.uint64)
        low = (ids & _WORD_MASK)[:, None]
        high = (ids >> np.uint64(_WORD_BITS))[:, None]
        grid = np.asarray(self.log_variance_grid, dtype=np.float64)
        classes = np.asarray(self.class_index)[rows]
        for component in np.unique(components):
            draws = np.flatnonzero(components == component)
            counter = np.asarray(draws, dtype=np.uint64)[None, :]
            words = philox4x32((low, high, counter, np.full_like(counter, MEMBER_STREAM)), self.key)
            node_uniform = _uniform(words[0], words[1], open_below=False)
            normal = ndtri(_uniform(words[2], words[3], open_below=True))
            omega = np.asarray(self.omega)[component, rows]
            shift = np.asarray(self.shift)[component, rows]
            values = np.empty((rows.shape[0], draws.shape[0]))
            for class_position in np.unique(classes):
                members = np.flatnonzero(classes == class_position)
                terms = _components(
                    np.asarray(self.log_density)[class_position], np.asarray(self.log_scale)[rows[members]], grid, omega[members], shift[members]
                )
                nodes = np.empty((members.shape[0], draws.shape[0]), dtype=np.int64)
                _sample_nodes(np.ascontiguousarray(terms.responsibility), np.ascontiguousarray(node_uniform[members]), nodes)
                conditional = np.take_along_axis(terms.conditional_variance, nodes, axis=1)
                values[members] = shift[members][:, None] * conditional + np.sqrt(conditional) * normal[members]
            out[:, draws] = values
        return out

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "class_index": np.asarray(self.class_index, dtype=np.int64), "log_scale": np.asarray(self.log_scale, dtype=np.float64),
            "omega": np.asarray(self.omega, dtype=np.float64), "log_density": np.asarray(self.log_density, dtype=np.float64),
            "log_variance_grid": np.asarray(self.log_variance_grid, dtype=np.float64), "shift": np.asarray(self.shift, dtype=np.float64),
            "weights": np.asarray(self.weights, dtype=np.float64),
            "key": np.asarray(self.key, dtype=np.uint64), "draw_count": np.asarray(int(self.draw_count), dtype=np.int64),
            "row_ids": np.arange(self.shape[0], dtype=np.int64) if self.row_ids is None else np.asarray(self.row_ids, dtype=np.int64),
        }

    def subset(self, rows: np.ndarray) -> "ProductMixtureDraws":
        """The law of ``rows`` (indices or a mask), drawing for each the numbers the whole law draws for it."""
        selected = np.arange(self.shape[0], dtype=np.int64)[rows]
        ids = selected if self.row_ids is None else np.asarray(self.row_ids, dtype=np.int64)[selected]
        return ProductMixtureDraws(
            class_index=np.asarray(self.class_index)[selected], log_scale=np.asarray(self.log_scale)[selected],
            omega=np.asarray(self.omega)[:, selected], log_density=self.log_density, log_variance_grid=self.log_variance_grid,
            shift=np.asarray(self.shift)[:, selected], weights=self.weights, key=self.key, draw_count=self.draw_count, row_ids=ids,
        )

    def __array__(self, dtype=None, copy=None) -> np.ndarray:
        """Every row of every draw (rows x K): the export path's whole matrix, never the scorer's."""
        return np.asarray(self.tile(0, self.shape[0]), dtype=dtype)


def law_from_arrays(kind: str, arrays: dict[str, np.ndarray]) -> DenseDraws | ProductMixtureDraws:
    """A law from ``DrawLaw.arrays`` (``artifact.load_model``)."""
    if kind == "dense":
        return DenseDraws(np.asarray(arrays["values"], dtype=np.float64))
    if kind == "product_mixture":
        key = np.asarray(arrays["key"], dtype=np.uint64)
        return ProductMixtureDraws(
            class_index=np.asarray(arrays["class_index"], dtype=np.int64), log_scale=np.asarray(arrays["log_scale"], dtype=np.float64),
            omega=np.asarray(arrays["omega"], dtype=np.float64), log_density=np.asarray(arrays["log_density"], dtype=np.float64),
            log_variance_grid=np.asarray(arrays["log_variance_grid"], dtype=np.float64), shift=np.asarray(arrays["shift"], dtype=np.float64),
            weights=np.asarray(arrays["weights"], dtype=np.float64),
            key=(int(key[0]), int(key[1])), draw_count=int(arrays["draw_count"]), row_ids=np.asarray(arrays["row_ids"], dtype=np.int64),
        )
    raise ValueError(f"no draw law {kind!r}")


def tile_rows(law: DrawLaw, working_bytes: int) -> int:
    """Rows per tile whose draws and working intermediates fit ``working_bytes`` (at least one)."""
    per_row = law.tile_row_bytes() + _FLOAT64_BYTES * law.shape[1]
    return max(1, int(working_bytes) // max(per_row, 1))


def export_draws(law: DrawLaw, path, working_bytes: int) -> None:
    """Write every draw (rows x K float64) to the ``.npy`` file ``path`` a tile at a time, so the export never holds
    more than a tile of rows (``tile_rows`` of ``working_bytes``)."""
    rows, count = law.shape
    target = np.lib.format.open_memmap(path, mode="w+", dtype=np.float64, shape=(rows, count))
    step = tile_rows(law, working_bytes)
    for start in range(0, rows, step):
        stop = min(start + step, rows)
        target[start:stop] = law.tile(start, stop)
    target.flush()
    del target
