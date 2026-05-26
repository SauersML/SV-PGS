"""Bitpacked GPU pipeline for SV-PGS.

Public entry point. CPU symbols and the GPU arch probe import eagerly; the
GPU kernel wrappers (``gemv_nt``, ``gemv_tn``, ``gemm_gram``, ``screen``) are
loaded lazily via PEP 562 ``__getattr__`` so this package can be imported in
environments without CuPy installed.

See ``BITPACKED_SPEC.md`` for the full contract.
"""

from __future__ import annotations

import importlib
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
# Each entry maps the public name on this package to
# ``(submodule_dotted_path, attr_inside_submodule)``. The submodule and the
# function inside it happen to share a name (e.g. submodule
# ``sv_pgs.bitpacked.gemv_nt`` exposes function ``gemv_nt``), so we MUST
# dereference the function attribute — otherwise ``bp.gemv_nt`` would resolve
# to the submodule itself and fail with "module object is not callable".
_LAZY_GPU_ATTRS: dict[str, tuple[str, str]] = {
    "gemv_nt": ("sv_pgs.bitpacked.gemv_nt", "gemv_nt"),
    "gemv_tn": ("sv_pgs.bitpacked.gemv_tn", "gemv_tn"),
    "gemm_gram": ("sv_pgs.bitpacked.gemm_gram", "gemm_gram"),
    "screen": ("sv_pgs.bitpacked.screening", "screen"),
}


def __getattr__(name: str) -> Any:
    """PEP 562 lazy attribute loader for GPU kernel wrappers."""
    if name in _LAZY_GPU_ATTRS:
        module_path, attr_name = _LAZY_GPU_ATTRS[name]
        try:
            module = importlib.import_module(module_path)
        except ImportError as exc:
            raise ImportError(
                f"Could not import {module_path!r} required for "
                f"attribute {name!r}: {exc}. This may indicate the "
                f"submodule has not yet been created, or that an "
                f"optional dependency (e.g. cupy) is missing."
            ) from exc
        try:
            value = getattr(module, attr_name)
        except AttributeError as exc:
            raise ImportError(
                f"Submodule {module_path} does not expose the expected "
                f"public symbol {attr_name!r}: {exc}."
            ) from exc
        # Cache the FUNCTION (not the submodule) for subsequent accesses.
        # NOTE: importing the submodule above set
        # ``sys.modules['sv_pgs.bitpacked'].<name> = <submodule>`` as a side
        # effect; we overwrite that here so future ``bp.<name>`` lookups
        # return the callable, not the submodule.
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_GPU_ATTRS))


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
