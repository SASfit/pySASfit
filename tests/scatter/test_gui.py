import os, numpy as np, pytest
from pysasfit.scatter import params, engine


def test_ini_roundtrip(tmp_path):
    p = params.defaults(); p["EditRadius"] = 12.5; p["LType"] = 5
    f = tmp_path / "a.ini"; params.write_ini(str(f), p, {"EditRadius": True}, "FCC Spheres")
    q, fit, sec = params.read_ini(str(f))
    assert q["EditRadius"] == 12.5 and q["LType"] == 5 and fit["EditRadius"] and sec == "FCC Spheres"


def test_engine_fcc_powder_and_1d():
    p = params.defaults(); p.update(LType=5, uca=40, EditRadius=10, EditSigma=0.05, EditQmax=1.0, GridPoints=32)
    im = engine.calc_2d(p)
    assert im.shape == (32, 32) and np.isfinite(im).all() and im.min() > 0
    q, I = engine.calc_1d(p)
    assert np.isfinite(I).all()


def test_engine_oriented_cylinder_symmetry():
    p = params.defaults(); p.update(ComboBoxParticle=1, Length=40, EditRadius=5, EditSigma=0.0, Ordis=0,
                                    EditDbeta=10, EditQmax=1.0, GridPoints=33, theta=90, phi=0)
    im = engine.calc_2d(p)
    assert np.allclose(im, im[::-1, ::-1], rtol=1e-6, atol=1e-9 * im.max())


def test_domain_width_convention():
    from pysasfit.scatter.lattices import domain_width, peak
    D = 400.0
    s = domain_width(D, "gaussian")
    x = np.linspace(-0.2, 0.2, 400001)
    L = peak(np.abs(x), s, "gaussian", d=1)
    half = x[L >= L.max() / 2]
    fwhm = half.max() - half.min()
    assert np.isclose(fwhm, np.sqrt(np.pi * np.log(2)) * 4 / D, rtol=1e-3)     # eq. 82
    assert np.isclose(np.trapezoid(L, x), 1.0, rtol=1e-6)
    sl = domain_width(D, "lorentzian")
    Ll = peak(np.abs(x), sl, "lorentzian", d=1)
    h = x[Ll >= Ll.max() / 2]
    assert np.isclose(h.max() - h.min(), 4 / D, rtol=2e-3)


def test_gamma_peak_limits():
    from pysasfit.scatter.lattices import gamma_peak
    x = np.linspace(-30, 30, 600001); d = 1.0
    for nu in (0.05, 1.0, 50.0):
        L = gamma_peak(x, d, nu)
        assert np.isclose(np.trapezoid(L, x), 1.0, atol=2e-3 if nu < 0.1 else 1e-6)
    lor = d / 2 / np.pi / (x ** 2 + (d / 2) ** 2)
    assert np.allclose(gamma_peak(x, d, 1e-3)[::1000][250:350], lor[::1000][250:350], rtol=2e-2)
    gau = 2 / (np.pi * d) * np.exp(-4 * x ** 2 / (np.pi * d ** 2))
    assert np.allclose(gamma_peak(x, d, 2000.0), gau, atol=2e-3 * gau.max())


def test_gamma_lamellae_normalisation_and_radial_powder():
    from pysasfit.scatter import Crystal
    c = Crystal("lamellae", 20.0, width=0.003, peak="gamma", nu=0.7, hmax=40)
    q = np.linspace(0.01, 6.0, 120001)
    z = c.Z(q, 0 * q, 0 * q)
    assert np.isclose(z.mean(), 1.0, rtol=0.03)
    a = Crystal("fcc", 40.0, width=0.004, peak="gaussian", hmax=4)
    b = Crystal("fcc", 40.0, width=0.004, peak="gaussian", hmax=4, powder_mode="radial")
    qq = np.linspace(0.15, 0.6, 400)
    za, zb = a.Z_powder(qq), b.Z_powder(qq)
    assert np.isclose(za.sum(), zb.sum(), rtol=0.02)


def test_gamma_lamellae_powder_matches_gaussian_limit():
    from pysasfit.scatter import Crystal
    g = Crystal("lamellae", 20.0, width=0.01, peak="gaussian", hmax=20)
    m = Crystal("lamellae", 20.0, width=0.01, peak="gamma", nu=3000.0, hmax=20)
    q = np.linspace(0.05, 3, 600)
    assert np.allclose(g.Z_powder(q), m.Z_powder(q), rtol=3e-3, atol=1e-4)
