"""Bitpacked GPU pipeline for SV-PGS.

Public entry point. CPU symbols and the GPU arch probe import eagerly; the
GPU kernel wrappers (``gemv_nt``, ``gemv_tn``, ``gemm_gram``, ``screen``) are
loaded lazily via PEP 562 ``__getattr__`` so this package can be imported in
environments without CuPy installed.

See ``BITPACKED_SPEC.md`` for the full contract.
"""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

# ---------------------------------------------------------------------------
# Eager imports: pure-Python / NumPy-only symbols.
# ---------------------------------------------------------------------------
# These modules must not import cupy at module load time.
from .lut import make_decode_lut
from .cpu_reference import (
    cpu_gemv_nt,
    cpu_gemv_tn,
    cpu_gemm_gram,
    cpu_screen,
)
from .launch import gpu_arch

# ---------------------------------------------------------------------------
# Lazy GPU kernel wrappers.
# ---------------------------------------------------------------------------
# Map of attribute name -> submodule name (relative to this package).
_LAZY_ATTRS: dict[str, str] = {
    "gemv_nt": "gemv_nt",
    "gemv_tn": "gemv_tn",
    "gemm_gram": "gemm_gram",
    "screen": "screening",
}


def __getattr__(name: str) -> Any:
    """PEP 562 lazy attribute loader for GPU kernel wrappers."""
    submod_name = _LAZY_ATTRS.get(name)
    if submod_name is None:
        raise AttributeError(
            f"module {__name__!r} has no attribute {name!r}"
        )
    try:
        submod = import_module(f"{__name__}.{submod_name}")
    except ImportError as exc:
        raise ImportError(
            f"Could not import sv_pgs.bitpacked submodule "
            f"{submod_name!r} required for attribute {name!r}: {exc}. "
            f"This may indicate the submodule has not yet been created, "
            f"or that an optional dependency (e.g. cupy) is missing."
        ) from exc
    try:
        attr = getattr(submod, name)
    except AttributeError as exc:
        raise ImportError(
            f"Submodule sv_pgs.bitpacked.{submod_name} does not expose "
            f"the expected public symbol {name!r}: {exc}."
        ) from exc
    globals()[name] = attr
    return attr


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_ATTRS))


if TYPE_CHECKING:  # pragma: no cover - type-checker hints only
    from .gemv_nt import gemv_nt
    from .gemv_tn import gemv_tn
    from .gemm_gram import gemm_gram
    from .screening import screen


__all__ = [
    "make_decode_lut",
    "cpu_gemv_nt",
    "cpu_gemv_tn",
    "cpu_gemm_gram",
    "cpu_screen",
    "gpu_arch",
    "gemv_nt",
    "gemv_tn",
    "gemm_gram",
    "screen",
]
