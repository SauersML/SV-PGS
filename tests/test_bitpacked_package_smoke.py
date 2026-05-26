from __future__ import annotations

import importlib

import pytest

pytest.importorskip("sv_pgs.bitpacked")


def test_bitpacked_module_imports() -> None:
    mod = importlib.import_module("sv_pgs.bitpacked")
    assert mod is not None
    assert hasattr(mod, "__name__")
    assert mod.__name__ == "sv_pgs.bitpacked"


def test_bitpacked_lazy_attribute_access() -> None:
    import sv_pgs.bitpacked as bp

    # cpu_gemv_nt is eagerly exported and must always be available.
    assert callable(bp.cpu_gemv_nt)

    # gemv_nt is lazy: either resolves (cupy installed) or raises a clear
    # AttributeError / ImportError. It MUST NOT raise NameError.
    try:
        attr = bp.gemv_nt
    except (AttributeError, ImportError):
        pass
    except NameError as exc:  # pragma: no cover - defensive
        pytest.fail(f"gemv_nt access raised NameError: {exc}")
    else:
        assert attr is not None


def test_make_decode_lut_via_package() -> None:
    import sv_pgs.bitpacked as bp

    lut = bp.make_decode_lut()
    assert lut.shape == (256, 4)
