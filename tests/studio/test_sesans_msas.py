"""SESANS and multiple small-angle scattering (MSAS) in pySASstudio: Hankel transforms, the .ses reader, the model,
the fitting engine and the explorer (no SASfit library needed: Python stand-ins for the sphere)."""

import os

import numpy as np
import pytest
from scipy.special import gammaln

from pysasfit.core.sesans_io import is_sesans_file, load_sesans
from pysasfit.studio.fitting import FitProblem
from pysasfit.studio.hankel import (MSASTransform, SESANSTransform, hankel_weights, log_grid, make_transform,
                                    sphere_sesans)
from pysasfit.studio.model import ModelFunction


@pytest.fixture(autouse=True)
def _builtin_hankel_defaults():
    """The tests use the built-in numerical defaults, not those saved in Settings → Integration… → Hankel transform."""
    from pysasfit.studio import hankel

    hankel.set_defaults(reset=True)
    yield
    hankel.set_defaults(reset=True)


# --------------------------------------------------------------------------- stand-ins
class _F:
    parameter_names = ["R", "eta"]


class Sphere:
    """(eta V F(qR))²: the contribution interface of the model and the fit."""

    sd = None

    def __init__(self, R=100.0, eta=1.0):
        self._v = {"ff": np.array([R, eta], float)}
        self._f = _F()

    def function_for(self, role):
        return self._f if role == "ff" else None

    def vector(self, role):
        return self._v.get(role)

    def amplitude_index(self):
        return None

    def intensity(self, q):
        R, eta = self._v["ff"]
        x = np.asarray(q, float) * R
        with np.errstate(all="ignore"):
            f = np.where(x > 1e-6, 3 * (np.sin(x) - x * np.cos(x)) / x**3, 1.0)
        return (eta * 4 / 3 * np.pi * R**3 * f) ** 2


class Gauss:
    """A exp(-q² a²): G(r) and the multiple-scattering series are known in closed form."""

    sd = None

    def __init__(self, A=1.0, a=30.0):
        self._v = {"ff": np.array([A, a], float)}
        self._f = _F()

    def function_for(self, role):
        return self._f if role == "ff" else None

    def vector(self, role):
        return self._v.get(role)

    def amplitude_index(self):
        return None

    def intensity(self, q):
        A, a = self._v["ff"]
        return A * np.exp(-np.asarray(q, float) ** 2 * a**2)


def gauss_msas(q, A, a, lam, t):
    """Multiple scattering of A exp(-q²a²): k0²/(2π t) Σ τ^n/n! (2a²/n) exp(-q²a²/n), τ = λ²t A/(4π a²)."""
    tau = lam**2 * t * A / (4 * np.pi * a**2)
    n = np.arange(1, 200)[:, None]
    terms = np.exp(n * np.log(tau) - gammaln(n + 1)) * (2 * a**2 / n) * np.exp(-np.asarray(q)[None, :] ** 2 * a**2 / n)
    return (2 * np.pi / lam) ** 2 / (2 * np.pi * t) * terms.sum(0), tau


# --------------------------------------------------------------------------- Hankel transforms
def test_weights_integrate_k_j0_exactly_for_linear_integrands():
    # f(k) k linear on [k0, kN], f constant below: ∫ f k J0(kx) dk of f = 1 is k J1(kx)/x up to kN
    from scipy.special import j1

    k = log_grid(1e-3, 5.0, 40)
    x = np.array([0.0, 0.5, 3.0, 40.0])
    W = hankel_weights(x, k, "j0")
    exact = np.where(x > 0, k[-1] * j1(k[-1] * x) / np.where(x > 0, x, 1), 0.5 * k[-1] ** 2)
    # f = 1 is not linear in f·k = k ... it is: g = k is linear, so the rule is exact
    assert np.allclose(W @ np.ones(k.size), exact, rtol=1e-9, atol=1e-12)


def test_sesans_of_a_gaussian():
    A, a = 2.0, 30.0
    d = np.geomspace(1, 1000, 50)
    m = ModelFunction([Gauss(A, a)], transform=SESANSTransform("G"))
    exact = A / (4 * np.pi * a**2) * (np.exp(-d**2 / (4 * a**2)) - 1)
    assert np.max(np.abs(m(d) - exact)) < 1e-3 * np.max(np.abs(exact))


def test_sesans_of_a_sphere_matches_the_analytic_projection():
    R = 1000.0
    d = np.geomspace(50, 4000, 80)
    m = ModelFunction([Sphere(R, 1.0)], scales=[1.0], transform=SESANSTransform("G"))
    G = m(d)
    G0 = (4 / 3 * np.pi * R**3) ** 2 * 9 / (4 * R**2) / (2 * np.pi)  # ∫ (V F)² q dq / 2π, ∫F² q dq = 9/(4R²)
    assert G[-1] == pytest.approx(-G0, rel=2e-3)
    assert np.max(np.abs(G / G0 - (sphere_sesans(d, R) - 1))) < 2e-3


def test_sphere_sesans_formula():
    from scipy.integrate import quad

    R = 1.0
    gamma = lambda r: (1 - 3 * r / (4 * R) + r**3 / (16 * R**3)) if r < 2 * R else 0.0  # noqa: E731
    proj = lambda d: quad(lambda z: gamma(np.hypot(d, z)), 0, 2 * R, limit=200)[0]  # noqa: E731
    for d in (0.2, 1.0, 1.9):
        assert sphere_sesans(np.array([d]), R)[0] == pytest.approx(proj(d) / proj(0.0), abs=1e-8)


def test_sesans_polarisation_uses_lambda_per_point():
    d = np.geomspace(100, 20000, 60)
    lam = np.linspace(2.0, 12.0, d.size)
    g = ModelFunction([Sphere(3000.0)], scales=[1e-14], transform=SESANSTransform("G"))(d)
    p = ModelFunction([Sphere(3000.0)], scales=[1e-14],
                      transform=SESANSTransform("P", thickness=0.2, wavelength=lam, wavelength_delta=d))(d)
    assert np.allclose(p, np.exp(lam**2 * 0.2 * g))
    sub = d[::7]  # other spin-echo lengths: λ interpolated
    p2 = ModelFunction([Sphere(3000.0)], scales=[1e-14],
                       transform=SESANSTransform("P", thickness=0.2, wavelength=lam, wavelength_delta=d))(sub)
    lp, lp2 = np.log(p[::7]), np.log(p2)  # the Q grid follows the δ range: tiny differences
    assert np.max(np.abs(lp2 - lp)) < 1e-3 * np.max(np.abs(lp))


@pytest.mark.parametrize("norm", ["transmission", "incident"])
@pytest.mark.parametrize("A", [5.0, 500.0])
def test_msas_of_a_gaussian_matches_the_series(norm, A):
    a, lam, t = 30.0, 6.0, 0.1
    q = np.geomspace(1e-3, 0.12, 80)
    tr = MSASTransform(t, lam, norm=norm)
    I = ModelFunction([Gauss(A, a)], transform=tr)(q)
    exact, tau = gauss_msas(q, A, a, lam, t)
    if norm == "incident":
        exact = exact * np.exp(-tau)
    assert tr.last_tau == pytest.approx(tau, rel=1e-3)
    ok = exact > 1e-2 * exact.max()
    assert np.max(np.abs(I - exact)[ok] / exact[ok]) < 3e-3
    assert np.max(np.abs(I - exact)) < 2e-4 * exact.max()  # the far tail: an absolute floor of the transform


def test_msas_tends_to_single_scattering_for_a_thin_sample():
    q = np.geomspace(0.003, 0.3, 60)
    single = ModelFunction([Sphere(50.0)], scales=[1e-9])(q)
    multi = ModelFunction([Sphere(50.0)], scales=[1e-9], transform=MSASTransform(1e-6, 0.6))(q)
    assert np.allclose(multi, single, rtol=1e-4)


def test_msas_refuses_an_absurd_scattering_probability():
    m = ModelFunction([Sphere(50.0)], scales=[1.0], transform=MSASTransform(1.0, 6.0))
    with pytest.raises(ValueError, match="too large"):
        m(np.geomspace(0.01, 0.3, 20))


def test_make_transform_names():
    assert make_transform("1") is None and make_transform(None) is None
    assert make_transform("H{...}/2pi").observable == "G"
    assert make_transform("Exp(H{...}/2pi)", thickness=0.2, wavelength=2.0).observable == "P"
    t = make_transform({"kind": "msas", "thickness": 0.3, "wavelength": 0.5, "norm": "transmission"})
    assert (t.thickness, t.wavelength, t.norm) == (0.3, 0.5, "transmission")
    assert make_transform(t.settings()).settings() == t.settings()
    with pytest.raises(ValueError):
        make_transform("hankel")


# --------------------------------------------------------------------------- model and fitting
def test_sesans_model_is_linear_and_its_parts_add_up():
    d = np.geomspace(50, 5000, 40)
    m = ModelFunction([Sphere(300.0), Sphere(900.0)], scales=[2e-10, 1e-12], background=0.01, bg_A=5.0,
                      transform=SESANSTransform("G"))
    assert m.is_linear and m.domain == "delta" and m.power_term(d) == 0.0
    parts, bg = m.components(d)
    assert np.allclose(m(d), sum(s * p for s, p in zip(m.scales, parts)) + bg)
    assert not ModelFunction([Sphere()], transform=MSASTransform()).is_linear
    with pytest.raises(ValueError):
        ModelFunction([Sphere()], transform=SESANSTransform("P")).components(d)


@pytest.mark.parametrize("vp", [False, True])
def test_fit_sesans_data(vp):
    d = np.geomspace(400, 47000, 120)
    G = ModelFunction([Sphere(10000.0)], scales=[1e-14], transform=SESANSTransform("G"))(d)
    s = np.full(d.shape, 0.004)
    y = G + np.random.default_rng(1).normal(0, s)
    p = FitProblem()
    p.add_dataset((d, y, s), ModelFunction([Sphere(8000.0)], scales=[2e-14], transform=SESANSTransform("G")))
    p.add_parameter("0.ff.R", lower=1000, upper=30000)
    p.add_parameter("scale.0", lower=0)
    p.add_parameter("background")
    r = p.fit("scipy", vp=vp)
    vals = {x.address: x.value for x in p.parameters}
    assert vals["0.ff.R"] == pytest.approx(10000, rel=0.01)
    assert r.chi2_red < 1.5
    if vp:
        assert "variable projection (2 linear" in r.message


def test_fit_msas_thickness_and_variable_projection_is_left_out():
    from scipy.integrate import quad

    G0 = quad(lambda x: Sphere(300.0).intensity(np.array([x]))[0] * x, 0, 1, limit=2000)[0] / (2 * np.pi)
    sc = 0.8 / (36 * 0.2 * G0)  # Σt = 0.8 for λ = 6, t = 0.2
    q = np.geomspace(1e-3, 0.05, 100)
    I = ModelFunction([Sphere(300.0)], scales=[sc], background=0.01, transform=MSASTransform(0.2, 6.0))(q)
    s = 0.01 * I
    y = I + np.random.default_rng(2).normal(0, s)
    p = FitProblem()
    p.add_dataset((q, y, s), ModelFunction([Sphere(280.0)], scales=[0.7 * sc], transform=MSASTransform(0.15, 6.0)))
    for a in ("0.ff.R", "scale.0", "background", "transform.thickness"):
        p.add_parameter(a, lower=0)
    assert p.linear_parameters() == []
    p.fit("scipy", vp=True)
    vals = {x.address: x.value for x in p.parameters}
    assert vals["transform.thickness"] == pytest.approx(0.2, rel=0.02)
    assert vals["0.ff.R"] == pytest.approx(300, rel=0.01)


# --------------------------------------------------------------------------- reader
SASVIEW = """FileFormatVersion       1.0
DataFileTitle           test
Thickness               0.2
Thickness_unit          cm
Theta_zmax              0.0168
Theta_zmax_unit         radians
SpinEchoLength_unit     A
Depolarisation_unit     A-2 cm-1
Wavelength_unit         A

BEGIN_DATA
SpinEchoLength Depolarisation Depolarisation_error SpinEchoLength_error Wavelength Wavelength_error Polarisation Polarisation_error
626.41	-0.0024819	0.0039245	31.32	2.11	0.1055	0.99779	0.0034868
391.56	0.0041929	0.0036894	19.578	2.11	0.1055	1.0037	0.0032974
47036	-0.20983	0.0049963	2351.8	2.11	0.1055	0.82958	0.0036907
"""

DELFT = """name\tse009051.dat
sample\t"SiO2 in D2O"
Thickness [mm]\t2
z-acceptance [nm^-1]\t0.3

spin echo length [um]\terror spin-echo length\twavelength [nm]\terror wavelength\tSample 5\tSample 5 -error
26\t5\t0.161245\t0.01\t0.996315\t0.00528445
28.08\t5\t0.167571\t0.01\t0.995937\t0.00522962
224.305\t5\t0.473608\t0.01\t0.90663\t0.0043316
"""


def test_read_sasview_ses(tmp_path):
    fn = tmp_path / "a.ses"
    fn.write_text(SASVIEW)
    assert is_sesans_file(fn)
    d = load_sesans(fn, length_unit="A")
    assert np.allclose(d.delta, [391.56, 626.41, 47036])  # sorted
    assert d.thickness == pytest.approx(0.2)
    assert np.allclose(d.G, [0.0041929, -0.0024819, -0.20983])
    assert np.allclose(d.P, [1.0037, 0.99779, 0.82958])
    assert d.theta_max == pytest.approx(0.0168) and d.q_accept is None
    n = load_sesans(fn, length_unit="nm")  # lengths / 10, G per nm² = 100 × per Å²
    assert np.allclose(n.delta, d.delta / 10) and np.allclose(n.G, d.G * 100) and np.allclose(n.wavelength, 0.211)


def test_read_delft_ses_with_corrected_units(tmp_path):
    fn = tmp_path / "b.ses"
    fn.write_text(DELFT)
    d = load_sesans(fn, length_unit="nm")
    assert d.file_units["delta"] == "um" and np.allclose(d.delta, [26000, 28080, 224305])
    d = load_sesans(fn, length_unit="nm", delta_unit="nm")  # the file says µm, but means nm
    assert np.allclose(d.delta, [26, 28.08, 224.305])
    assert d.thickness == pytest.approx(0.2) and d.q_accept == pytest.approx(0.3)
    assert d.columns["P"] == "Sample 5"
    assert np.allclose(d.G, np.log(d.P) / (d.wavelength**2 * 0.2))  # derived G(δ) - G(0)
    assert np.allclose(d.dG, d.dP / (d.P * d.wavelength**2 * 0.2))


# --------------------------------------------------------------------------- explorer
@pytest.fixture
def explorer(tmp_path, monkeypatch):
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets

    from pysasfit.studio import register_python_function, unregister_python_function

    def sph(q, p):
        return Sphere(p[0], p[1]).intensity(q)

    monkeypatch.setenv("PYSASFIT_SETTINGS", str(tmp_path / "settings.json"))
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    register_python_function("ff_pysphere", sph, parameters=("R", "eta"), defaults=(5.0, 1.0))
    from pysasfit.studio.explorer import ModelExplorer

    e = ModelExplorer(ff="ff_pysphere")
    yield e, app
    e.close()
    unregister_python_function("ff_pysphere")


def _ses_file(tmp_path, R=10000.0, sc=2.6e-18):  # G(∞) - G(0) ≈ -0.2 Å⁻²cm⁻¹ as the polystyrene data
    d = np.geomspace(400, 47000, 80)
    G = ModelFunction([Sphere(R)], scales=[sc], transform=SESANSTransform("G"))(d)
    G = G + np.random.default_rng(3).normal(0, 0.003, d.size)
    P = np.exp(2.11**2 * 0.2 * G)
    lines = ["Thickness [cm]: 2.000000e-01", "",
             "spin echo length [A]\tDepolarisation [A-2 cm-1]\terror depol [A-2 cm-1]\terror SEL [A]\twavelength [A]\t"
             "error wavelength [A]\tpolarisation\terror pol"]  # fmt: skip
    for di, gi, pi in zip(d, G, P):
        lines.append(f"{di:.6g}\t{gi:.6g}\t0.003\t{di * 0.05:.4g}\t2.11\t0.1\t{pi:.6g}\t{pi * 2.11**2 * 0.2 * 0.003:.4g}")
    fn = tmp_path / "ps.ses"
    fn.write_text("\n".join(lines) + "\n")
    return str(fn)


def _start(e, R, scale):
    p = e.models()[0]
    rows = {r.name: r for r in p.param_rows["ff"]}
    rows["R"].set_value(R)
    rows["eta"].set_value(1.0)
    p.scale_row.set_value(scale)
    return p, rows


def test_explorer_fits_a_sesans_file_and_switches_the_axis(explorer, tmp_path):
    e, _app = explorer
    e.units.setCurrentIndex(e.units.findText("1/Å"))
    e.fit_panel.load_files([_ses_file(tmp_path)], sesans_options={"observable": "G"})
    d = e.datasets[0]
    assert d["domain"] == "delta" and e._shown_domain == "delta" and d["transform"]["kind"] == "sesans"
    assert d["transform"]["thickness"] == pytest.approx(0.2) and not e.logy.isChecked()
    p, rows = _start(e, 9000.0, 2e-18)
    e._force_calc()
    assert "spin-echo length" in (e.ax_res.get_xlabel() or e.ax.get_xlabel()) and e.ax.get_ylabel().startswith("G")
    rows["R"].set_fit(True)
    p.scale_row.set_fit(True)
    prob, _m = e.fit_panel._build_problem()
    prob.fit("scipy", vp=True)
    assert {x.address: x.value for x in prob.parameters}["0.ff.R"] == pytest.approx(10000, rel=0.02)
    # P/P0: the data change with the transform
    e.set_transform("sesans_p")
    assert d["observable"] == "P" and np.all(d["I"] > 0.5) and e.ax.get_ylabel() == "P/P0"
    with pytest.raises(ValueError):
        e.set_transform("msas")  # not for spin-echo lengths


def test_explorer_msas_curve_project_and_python_code(explorer, tmp_path):
    e, _app = explorer
    p, rows = _start(e, 30.0, 2e-6)
    e.set_transform("msas", thickness=0.2, wavelength=0.6)
    e._force_calc()
    assert "Σt" in e.status.text() and e.single_line.get_visible()
    q = np.geomspace(0.01, 0.5, 60)
    I = e.model_for(None)(q)
    e.add_dataset(q, I, 0.01 * I, "msas.dat")
    assert e.datasets[0]["transform"]["kind"] == "msas"  # the new data set takes the transform shown
    e.fit_panel.load_files([_ses_file(tmp_path)], sesans_options={"observable": "G", "delta_unit": "nm"})
    assert e.current_ds == 1 and e._shown_domain == "delta"
    e.ds_combo.setCurrentIndex(0)
    assert e._shown_domain == "q" and e.transform_box.kind.currentData() == "msas"
    path = e.save_project(str(tmp_path / "p.sasstudio")) and e.project_path
    from pysasfit.studio.explorer import ModelExplorer

    e2 = ModelExplorer(ff="ff_pysphere")
    try:
        assert e2.open_project(path)
        assert [x["domain"] for x in e2.datasets] == ["q", "delta"]
        assert e2._tr_settings(0)["kind"] == "msas" and e2._tr_settings(1)["kind"] == "sesans"
        assert np.allclose(e2.datasets[1]["sesans"]["lam"], e.datasets[1]["sesans"]["lam"])
    finally:
        e2.close()
    code = e.python_code()
    assert "make_transform('msas'" in code and "load_sesans(" in code
    compile(code, "generated", "exec")


# --------------------------------------------------------------------------- resolution (acceptance)
def _clipped_reference(A, a, d, qc):
    from scipy.integrate import quad
    from scipy.special import j0

    return quad(lambda q: A * np.exp(-q * q * a * a) * q * (j0(q * d) - 1), 0, qc, limit=500)[0] / (2 * np.pi)


def test_sesans_acceptance_clips_the_hankel_transform():
    A, a = 2.0, 30.0
    d = np.geomspace(5, 1000, 30)
    G = ModelFunction([Gauss(A, a)], transform=SESANSTransform("G", q_accept=0.02))(d)
    ref = np.array([_clipped_reference(A, a, x, 0.02) for x in d])
    assert np.max(np.abs(G - ref)) < 5e-4 * np.max(np.abs(ref))


def test_sesans_acceptance_per_point_from_theta_and_lambda():
    A, a = 2.0, 30.0
    d = np.geomspace(5, 1000, 30)
    lam = np.linspace(2.0, 12.0, d.size)  # time of flight: λ grows with δ
    th = 0.005
    T = SESANSTransform("G", theta_max=th, wavelength=lam, wavelength_delta=d)
    qc = 2 * np.pi * np.sin(th) / lam
    assert np.allclose(T.q_max(d), qc)
    G = ModelFunction([Gauss(A, a)], transform=T)(d)
    ref = np.array([_clipped_reference(A, a, x, q) for x, q in zip(d, qc)])
    assert np.max(np.abs(G - ref)) < 1e-4 * np.max(np.abs(ref))
    same = SESANSTransform("G", q_accept=qc, accept_delta=d)  # the same limits given as an array
    assert np.allclose(ModelFunction([Gauss(A, a)], transform=same)(d), G)
    assert make_transform(T.settings()).settings() == T.settings()


def test_hankel_weights_with_a_uniform_limit_equal_a_shorter_grid():
    k = log_grid(1e-3, 1.0, 32)
    x = np.array([0.0, 3.0, 50.0])
    cut = k[60]
    W = hankel_weights(x, k, "j0m1", True, 4.0, k_upper=cut)
    W2 = hankel_weights(x, k[:61], "j0m1", True, None)
    assert np.allclose(W[:, :61], W2) and np.all(W[:, 61:] == 0)


def test_explorer_sesans_resolution(explorer, tmp_path):
    e, _app = explorer
    e.units.setCurrentIndex(e.units.findText("1/Å"))
    fn = tmp_path / "sv.ses"
    fn.write_text(SASVIEW)
    e.fit_panel.load_files([str(fn)], sesans_options={"observable": "G"})
    s = e._tr_settings(0)
    assert s["accept"] == "theta" and s["theta_max"] == pytest.approx(0.0168)
    assert e.transform_box.accept.currentData() == "theta"
    t = e.transform_for(0)
    assert np.allclose(t.q_max(e.datasets[0]["q"]), 2 * np.pi * np.sin(0.0168) / 2.11)
    e.set_transform(accept="q", q_accept=0.01)
    assert e.transform_for(0).q_accept == pytest.approx(0.01) and e.transform_for(0).theta_max is None
    e.set_transform(accept="none")
    assert e.transform_for(0).q_max(e.datasets[0]["q"]) is None
    e.set_transform(accept="theta")  # the angle of the file is still known
    assert e.transform_for(0).theta_max == pytest.approx(0.0168)
    assert "theta_max=0.0168" in e.python_code()


# --------------------------------------------------------------------------- FFTLog and QDHT (SASfit's strategies)
@pytest.mark.filterwarnings("ignore:SESANS QDHT")  # 3 decades of δ: the warning is right, the Gaussian is 0 there
@pytest.mark.parametrize("method, tol", [("fftlog", 1e-4), ("qdht", 1e-4)])
def test_sesans_fftlog_and_qdht(method, tol):
    A, a = 2.0, 30.0
    d = np.geomspace(1, 1000, 50)
    exact = A / (4 * np.pi * a**2) * (np.exp(-d**2 / (4 * a**2)) - 1)
    G = ModelFunction([Gauss(A, a)], transform=SESANSTransform("G", method=method))(d)
    assert np.max(np.abs(G - exact)) < tol * np.max(np.abs(exact))
    R = 1000.0
    ds = np.geomspace(50, 4000, 60)
    G0 = (4 / 3 * np.pi * R**3) ** 2 * 9 / (4 * R**2) / (2 * np.pi)
    Gs = ModelFunction([Sphere(R)], transform=SESANSTransform("G", method=method))(ds)
    assert np.max(np.abs(Gs / G0 - (sphere_sesans(ds, R) - 1))) < tol
    # with the resolution: the node sum is cut inside the cell of Q_max
    Gc = ModelFunction([Gauss(A, a)], transform=SESANSTransform("G", method=method, q_accept=0.02))(d)
    ref = np.array([_clipped_reference(A, a, x, 0.02) for x in d])
    assert np.max(np.abs(Gc - ref)) < 5e-3 * np.max(np.abs(ref))


@pytest.mark.parametrize("method", ["fftlog", "qdht"])
@pytest.mark.parametrize("A", [5.0, 500.0])
def test_msas_fftlog_and_qdht(method, A):
    a, lam, t = 30.0, 6.0, 0.1
    q = np.geomspace(1e-3, 0.12, 80)
    tr = MSASTransform(t, lam, norm="incident", method=method)
    I = ModelFunction([Gauss(A, a)], transform=tr)(q)
    exact, tau = gauss_msas(q, A, a, lam, t)
    exact = exact * np.exp(-tau)
    ok = exact > 1e-2 * exact.max()
    assert np.max(np.abs(I - exact)[ok] / exact[ok]) < 1e-4
    assert tr.last_tau == pytest.approx(tau, rel=1e-6)


def test_methods_agree_on_polydisperse_spheres_and_keep_their_settings():
    class Poly(Sphere):
        def intensity(self, q):
            Rs = np.linspace(20, 60, 41)
            w = np.exp(-0.5 * ((Rs - 40) / 6) ** 2)
            return sum(wi * Sphere(R).intensity(q) for R, wi in zip(Rs, w / w.sum()))

    q = np.geomspace(0.01, 2.0, 60)
    out = {}
    for m in ("filon", "fftlog", "qdht"):
        tr = MSASTransform(1.0, 0.6, method=m)
        out[m] = ModelFunction([Poly()], scales=[2e-10], transform=tr)(q)
        assert make_transform(tr.settings()).settings() == tr.settings()
    assert np.max(np.abs(out["fftlog"] / out["qdht"] - 1)) < 1e-3
    assert np.max(np.abs(out["filon"] / out["qdht"] - 1)) < 5e-3
    with pytest.raises(ValueError):
        MSASTransform(method="ogata")


def test_explorer_hankel_method(explorer):
    e, _app = explorer
    _start(e, 30.0, 2e-6)
    e.set_transform("msas", thickness=0.2, wavelength=0.6, method="qdht", n_points=1024)
    t = e.transform_for(None)
    assert t.method == "qdht" and t.n_points == 1024
    assert e.transform_box.method.currentData() == "qdht" and e.transform_box.n_points.isVisible() is not None
    e._force_calc()
    assert "Σt" in e.status.text()
    assert "method='qdht'" in e.python_code() and "n_points=1024" in e.python_code()


def test_hankel_defaults_in_the_integration_dialog(explorer):
    from pysasfit.studio import hankel
    from pysasfit.studio.integration_dialog import IntegrationDialog
    from pysasfit.studio.library import DEFAULT_PRECISION

    e, _app = explorer
    dlg = IntegrationDialog(dict(DEFAULT_PRECISION), dict(e.integration), hankel=hankel.defaults())
    dlg.h_method.setCurrentIndex(dlg.h_method.findData("qdht"))
    dlg.h_fields["n_points"].setText("1024")
    dlg.h_fields["sesans_per_decade"].setText("80")
    dlg.accept()
    h = dlg.hankel_result()
    assert h["method"] == "qdht" and h["n_points"] == 1024 and h["sesans_per_decade"] == 80
    dlg2 = IntegrationDialog(dict(DEFAULT_PRECISION), dict(e.integration), hankel=hankel.defaults())
    dlg2.h_fields["tail_power"].setText("1.5")
    dlg2.accept()
    assert "tail_power" in dlg2.error.text()  # refused, the dialog stays open
    # the explorer uses them for every transform that does not choose its own method
    _start(e, 30.0, 2e-6)
    e.set_transform("msas", thickness=0.2, wavelength=0.6)
    assert e.transform_for(None).method == "filon"
    e._apply_integration({}, dict(e.integration), save=False, hankel_settings=h)
    t = e.transform_for(None)
    assert t.method == "qdht" and t.n_points == 1024
    e.set_transform(method="fftlog")  # a data set may choose its own
    assert e.transform_for(None).method == "fftlog"
    assert SESANSTransform("G").per_decade == 80 and SESANSTransform("G", per_decade=10).per_decade == 10


def test_hankel_settings_are_saved(tmp_path, monkeypatch):
    from pysasfit.studio import hankel, paths

    store = {}
    monkeypatch.setattr(paths, "get_setting", lambda key, default=None: store.get(key, default))
    monkeypatch.setattr(paths, "set_setting", lambda key, value: store.__setitem__(key, value))
    hankel.save_settings({**hankel.DEFAULTS, "method": "fftlog", "msas_r_per_decade": 512})
    got = hankel.load_settings()
    assert got["method"] == "fftlog" and got["msas_r_per_decade"] == 512 and got["n_points"] == 2048
    store["hankel"]["method"] = "nonsense"
    assert hankel.load_settings()["method"] == "filon"
    with pytest.raises(KeyError):
        hankel.set_defaults(points=3)


def test_fftlog_plan_fast_lengths_and_reuse():
    from scipy import fft

    from pysasfit.studio.hankel import FFTLogPlan, fast_lengths, fftlog_grid, is_fast_length

    assert fast_lengths(2047) == (2025, 2048) and fast_lengths(2048) == (2048, 2048) and not is_fast_length(4097)
    k, r, dln, off = fftlog_grid(1e-4, 10.0, 512)
    plan = FFTLogPlan.get(512, dln, off)
    assert FFTLogPlan.get(512, dln, off) is plan  # shared
    a = np.random.default_rng(0).random((3, 512))
    ref = fft.fht(a, dln, mu=0.0, offset=off)
    assert np.allclose(plan.apply(a), ref) and np.allclose(plan.matrix() @ a[0], ref[0])
    _k, _r, dln2, off2 = fftlog_grid(1e-4, 10.0, 2047)
    with pytest.warns(UserWarning, match="2025 and 2048"):
        slow = FFTLogPlan(2047, dln2, off2)  # suggested, not taken
    assert slow.n == 2047


def test_slow_fft_length_is_only_suggested_in_the_dialog(explorer):
    from pysasfit.studio import hankel
    from pysasfit.studio.integration_dialog import IntegrationDialog
    from pysasfit.studio.library import DEFAULT_PRECISION

    e, _app = explorer
    dlg = IntegrationDialog(dict(DEFAULT_PRECISION), dict(e.integration), hankel=hankel.defaults())
    dlg.h_fields["n_points"].setText("2047")
    assert "2025 or 2048" in dlg.h_nhint.text()
    dlg.accept()
    assert dlg.hankel_result()["n_points"] == 2047
    dlg.h_fields["n_points"].setText("4096")
    assert dlg.h_nhint.text() == ""


# --------------------------------------------------------------------------- libsasfit's sasfit_hankel
# The mock libsasfit integrates with a plain Simpson rule: these tests check the plumbing (callbacks, G(0), the
# resolution branch, the MSAS legs), not SASfit's strategies.
def test_library_hankel_wrapper(studio_lib):
    a = 30.0
    x = np.array([0.0, 10.0, 50.0])
    got = studio_lib.hankel(lambda k: np.exp(-k * k * a * a), x)
    assert np.allclose(got, np.exp(-x**2 / (4 * a * a)) / (2 * a * a), rtol=1e-6)

    def bad(k):
        raise ZeroDivisionError("inside the callback")

    with pytest.raises(ZeroDivisionError):
        studio_lib.hankel(bad, [1.0])


@pytest.mark.parametrize("source", ["spline", "direct"])
def test_sesans_with_sasfit_hankel(studio_lib, source):
    A, a = 2.0, 30.0
    d = np.geomspace(1, 1000, 8)
    exact = A / (4 * np.pi * a**2) * (np.exp(-d**2 / (4 * a**2)) - 1)
    t = SESANSTransform("G", method="sasfit", sasfit_source=source)
    assert not t.linear  # no weight matrix: variable projection leaves the scales to the optimiser
    G = ModelFunction([Gauss(A, a)], transform=t)(d)
    assert np.max(np.abs(G - exact)) < 1e-5 * np.max(np.abs(exact))
    Gc = ModelFunction([Gauss(A, a)], transform=SESANSTransform("G", method="sasfit", q_accept=0.02))(d)
    ref = np.array([_clipped_reference(A, a, x, 0.02) for x in d])
    assert np.max(np.abs(Gc - ref)) < 1e-4 * np.max(np.abs(ref))


def test_msas_with_sasfit_hankel(studio_lib):
    a, lam, t = 30.0, 6.0, 0.1
    q = np.geomspace(1e-3, 0.12, 8)
    exact, tau = gauss_msas(q, 500.0, a, lam, t)
    tr = MSASTransform(t, lam, norm="transmission", method="sasfit", q_range=(1e-4, 1.0), r_min_factor=0.3,
                       r_max_factor=0.3, sasfit_r_per_decade=12)  # ranges the mock's Simpson rule can resolve
    I = ModelFunction([Gauss(500.0, a)], transform=tr)(q)
    assert np.max(np.abs(I - exact) / exact) < 1e-3
    assert make_transform(tr.settings()).settings() == tr.settings()
