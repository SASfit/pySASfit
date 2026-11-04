"""Beam profile (2D Gaussian fit to the direct beam), divergence and the q resolution in 1D and 2D."""
import os

import h5py
import numpy as np
import pytest
from scipy.special import ndtr

from pysasfit.sasred.core.beamprofile import BeamProfile, beam_profiles, fit_beam_profile
from pysasfit.sasred.core.geometry import pixel_geometry
from pysasfit.sasred.core.resolution import resolution_2d, sigma_lambda_rel, sigma_q_1d

RAW = os.environ.get("PYSASRED_TEST_RAW", "/mnt/user-data/uploads/Sept2026/raw")
C8 = "SD 8.0 m, λ 6.00 Å, coll 8.0 m"
C2 = "SD 2.0 m, λ 6.00 Å, coll 8.0 m"


def _gauss_image(x0=60.3, y0=64.7, sx=1.6, sy=2.4, A=2e5, B=3.0, n=128, seed=4):
    i = np.arange(n, dtype=float)
    gx = ndtr((i + 0.5 - x0) / sx) - ndtr((i - 0.5 - x0) / sx)
    gy = ndtr((i + 0.5 - y0) / sy) - ndtr((i - 0.5 - y0) / sy)
    lam = A * gy[:, None] * gx[None, :] + B
    return np.random.default_rng(seed).poisson(lam).astype(float)


def test_fit_recovers_pixel_integrated_gaussian():
    f = fit_beam_profile(_gauss_image())
    assert f["x0"] == pytest.approx(60.3, abs=0.01) and f["y0"] == pytest.approx(64.7, abs=0.01)
    assert f["sigma_x"] == pytest.approx(1.6, rel=0.01) and f["sigma_y"] == pytest.approx(2.4, rel=0.01)
    assert f["chi2"] < 1.5
    # narrower than a pixel: still the width of the beam itself, without the pixel
    g = fit_beam_profile(_gauss_image(sx=0.4, sy=0.5))
    assert g["sigma_x"] == pytest.approx(0.4, rel=0.05) and g["sigma_y"] == pytest.approx(0.5, rel=0.05)


def _profile(tx, ty, SD=8.0, px=7.5e-3):
    return BeamProfile((SD, 6.0, 8.0), 64, 64, tx * SD / px, ty * SD / px, pixel_size_m=(px, px), SD_m=SD,
                       wavelength_A=6.0, runs=[1])


def test_geometric_resolution_formula():
    geo = pixel_geometry((128, 128), 7.5e-3, (63.5, 63.5), 8.0, 6.0)
    p = _profile(1.5e-3, 1.0e-3)
    assert p.divergence_rad == pytest.approx((1.5e-3, 1.0e-3))
    from pysasfit.sasred.core.averaging import q_edges, radial_binning
    b = radial_binning(geo.q, np.ones(geo.q.shape, bool), q_edges(0.01, 0.05, 8))
    s, parts = sigma_q_1d(b, geo, 0.10, "triangular", p)
    k = 2 * np.pi / 6.0
    # isotropic average of the geometric term: k^2 (tx^2 + ty^2)/2 (cos theta ~ 1)
    assert parts["geometry"] == pytest.approx(k ** 2 * (1.5e-3 ** 2 + 1e-3 ** 2) / 2, rel=0.03)
    assert parts["wavelength"] == pytest.approx((b.q_mean * 0.10 / np.sqrt(6)) ** 2)
    assert np.allclose(s ** 2, parts["wavelength"] + parts["bin"] + parts["geometry"])
    r = resolution_2d(geo, 0.10, "triangular", p, 7.5e-3, 8.0)
    # a pixel on the +x axis: parallel = x direction
    iy, ix = 64, 120
    c2 = (k * np.cos(0.5 * geo.two_theta[iy, ix])) ** 2
    w = (geo.q[iy, ix] * sigma_lambda_rel(0.10)) ** 2
    pix = c2 * (7.5e-3 / 8.0) ** 2 / 12
    assert r["qx_dev"][iy, ix] ** 2 == pytest.approx(c2 * 1.5e-3 ** 2 + pix + w * np.cos(geo.phi[iy, ix]) ** 2, rel=1e-3)
    assert r["q_perpendicular_dev"][iy, ix] ** 2 == pytest.approx(c2 * 1e-3 ** 2 + pix, rel=0.01)
    assert np.all(np.abs(r["rho_xy"]) <= 1)


@pytest.fixture(scope="module")
def vifor():
    if not os.path.isdir(RAW):
        pytest.skip("Vifor raw data not available")
    from pysasfit.sasred import plan as pl, reduction_table as rt
    from pysasfit.sasred.catalogue import beam_centres, scan_folder
    from pysasfit.sasred.io.sinq import read_sans_i
    cat = scan_folder(RAW)
    raws = {e.run: e.raw for e in cat if e.raw}
    plan = pl.build_plan(cat)
    read = (lambda n: read_sans_i(os.path.join(RAW, f"sans2026n{n:06d}.hdf"), checksum=False))
    return cat, raws, plan, beam_centres(list(raws.values())), read


def test_profiles_of_the_vifor_geometries(vifor):
    cat, raws, *_ = vifor
    ps = beam_profiles(list(raws.values()))
    p8 = ps[(8.0, 6.0, 8.0)]
    assert p8.source == "empty beam" and p8.runs == [82173, 82221, 82269]
    assert 10 < p8.sigma_x_mm < 17 and p8.sigma_y_mm == pytest.approx(p8.sigma_x_mm, rel=0.05)
    assert (2.0, 6.0, 8.0) not in ps                      # no direct beam at 2 m: no resolution there
    assert "divergence" in p8.describe()


def test_row_reduction_with_and_without_resolution(vifor, tmp_path):
    from pysasfit.sasred import absolute as ab, reduction_table as rt
    from pysasfit.sasred.core.averaging import q_edges
    from pysasfit.sasred.io.ascii import write_ascii_1d
    from pysasfit.sasred.io.nxcansas import write_nxcansas_1d, write_nxcansas_2d
    cat, raws, plan, cs, read = vifor
    t = rt.from_plan(plan, C8, raws)
    c = cs[(8.0, 6.0, 8.0)]
    prof = beam_profiles(list(raws.values()))[(8.0, 6.0, 8.0)]
    e = q_edges(0.006, 0.09, 30)
    r2, r1 = ab.reduce_row(t, t.samples[2], read, (c.x, c.y), e, beam_profile=prof)
    assert r1.q_sigma is not None and np.all(r1.q_sigma > 0)
    assert any(n.startswith("Qdev: wavelength") and "divergence" in n for n in r1.notes)
    # at the lowest q the divergence dominates, at high q the wavelength
    s_l = sigma_lambda_rel(r2.raw.dlam_over_lam_fwhm)
    assert r1.q_sigma[0] > 2 * r1.q[0] * s_l and r1.q_sigma[-1] == pytest.approx(r1.q[-1] * s_l, rel=0.3)
    assert r2.resolution is not None and r2.resolution["qx_dev"].shape == r2.image.shape
    write_nxcansas_2d(tmp_path / "a2d.h5", r2)
    with h5py.File(tmp_path / "a2d.h5") as f:
        d = f["sasentry01/sasdata01"]
        assert d["Qx"].attrs["resolutions"] in (b"Qxdev", "Qxdev") and d["Qxdev"].shape == r2.image.shape
    # without the profile: reduced, no Qdev, said so; files without / with zero Qdev
    r2n, r1n = ab.reduce_row(t, t.samples[2], read, (c.x, c.y), e)
    assert r1n.q_sigma is None and r2n.resolution is None
    assert any("resolution (Qdev) not determined" in n for n in r1n.notes)
    assert np.allclose(r1n.I, r1.I)
    write_nxcansas_1d(tmp_path / "n.h5", r1n)
    with h5py.File(tmp_path / "n.h5") as f:
        d = f["sasentry01/sasdata01"]
        assert "Qdev" not in d and "resolutions" not in d["Q"].attrs
    write_ascii_1d(tmp_path / "n.dat", r1n)
    txt = open(tmp_path / "n.dat").read()
    assert "Qdev = 0: resolution not determined" in txt
    assert np.all(np.loadtxt(tmp_path / "n.dat")[:, 3] == 0)


def test_gui_reports_missing_resolution():
    """2 m has no direct beam: the divergence is estimated from the collimation, said in the log and in
    'Missing input'; 8 m uses the fitted profile."""
    if not os.path.isdir(RAW):
        pytest.skip("Vifor raw data not available")
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from pysasfit.sasred.gui import MainWindow
    w = MainWindow()
    w.set_folder(RAW)
    log = w.log_view.toPlainText()
    assert "beam profile coll 8 m, SD 8 m" in log and "divergence estimated from the collimation" in log
    pages = {p.table.config: p for p in w.table_panel.pages()}
    t8, t2 = pages[C8].table, pages[C2].table
    t8.samples, t2.samples = t8.samples[:1], t2.samples[:1]
    tag = w.reduce_table_rows(t8, t8.samples)
    assert w.results[tag].q_sigma is not None
    assert any("from the direct beam" in n for n in w.results[tag].notes)
    tag2 = w.reduce_table_rows(t2, t2.samples)
    assert w.results[tag2].q_sigma is not None
    assert any("estimated from the collimation (calibrated on the direct beam of coll 8 m, runs 82173" in n
               for n in w.results[tag2].notes)
    items = [w.missing_view.topLevelItem(i) for i in range(w.missing_view.topLevelItemCount())]
    assert any(it.text(2) == "resolution" and "estimated from the" in it.text(3) for it in items)
    w.close()


def test_collimation_estimate():
    """σ_spot² = (a1 SD/coll)²/12 + (a2 (coll+SD)/coll)²/12 per axis; close to the measured 8 m beam."""
    from types import SimpleNamespace as NS
    from pysasfit.sasred.core.beamprofile import collimation_profile
    inst = {"collimation": {"source_aperture_mm": [50, 50], "sample_aperture_mm": [8, 14]}}
    r = NS(coll_m=8.0, SD_m=8.0, SD_effective_m=8.0, wavelength_A=6.0, pixel_size_m=(7.5e-3, 7.5e-3), L=None)
    p = collimation_profile(r, inst)
    sx = np.sqrt((50 * 1.0) ** 2 / 12 + (8 * 2.0) ** 2 / 12)
    sy = np.sqrt((50 * 1.0) ** 2 / 12 + (14 * 2.0) ** 2 / 12)
    assert p.estimated and p.sigma_x_mm == pytest.approx(sx) and p.sigma_y_mm == pytest.approx(sy)
    assert p.divergence_rad[0] == pytest.approx(sx * 1e-3 / 8.0)
    assert "estimated from the collimation (guide 50×50 mm², sample aperture 8×14 mm²" in p.origin()
    assert collimation_profile(r, {}) is None                                   # no apertures known
    assert collimation_profile(NS(**dict(vars(r), coll_m=0.0)), inst) is None   # no collimation length
    if os.path.isdir(RAW):                                                      # measured 8 m: σ = 13.6 mm
        from pysasfit.sasred.io.sinq import read_sans_i
        est = collimation_profile(read_sans_i(os.path.join(RAW, "sans2026n082173.hdf"), checksum=False))
        assert 13.6 < est.sigma_x_mm < 1.3 * 13.6


def test_dlam_from_selector_tilt():
    from pysasfit.sasred.core.resolution import dlam_from_tilt
    from pysasfit.sasred.reduce import load_instrument
    t = load_instrument("sans_i")["wavelength"]["selector_tilt"]
    for x, d in zip([0, -2, -4, -6, -8, -10], [0.09832, 0.1081, 0.12666, 0.1579, 0.20619, 0.27177]):
        assert dlam_from_tilt(x, t) == (pytest.approx(d), "")
    v = [dlam_from_tilt(x, t)[0] for x in np.linspace(-10, 0, 41)]
    assert np.all(np.diff(v) < 0)                          # monotone between the points
    assert dlam_from_tilt(0.02, t) == (pytest.approx(0.09832), "")      # read-back noise
    d, note = dlam_from_tilt(-12, t)
    assert d == pytest.approx(0.27177) and "outside the calibration table" in note


def test_raw_run_takes_dlam_from_tilt():
    if not os.path.isdir(RAW):
        pytest.skip("Vifor raw data not available")
    from pysasfit.sasred.io.sinq import read_sans_i
    r = read_sans_i(os.path.join(RAW, "sans2026n082173.hdf"), checksum=False)
    assert abs(r.selector_tilt_deg) < 0.01
    assert r.dlam_over_lam_fwhm == pytest.approx(0.09832) and r.dlam_source.startswith("selector tilt 0.00°")


def test_mixed_selector_tilts_are_flagged(vifor):
    """Water at another tilt: fine (info) with the monitor normalisation, to check with the time one."""
    import copy
    from pysasfit.sasred import absolute as ab, reduction_table as rt
    cat, raws, plan, cs, read = vifor
    t = rt.from_plan(plan, C8, raws)
    c = cs[(8.0, 6.0, 8.0)]
    water = set(t.corrections["water"].used)

    def read_tilted(n):
        r = read(n)
        if n in water:
            r = copy.copy(r)
            r.selector_tilt_deg = -4.0
        return r
    from pysasfit.sasred.reduce import load_instrument
    behind = load_instrument("sans_i")
    behind["monitor"]["normaliser_after_selector"] = True
    front = load_instrument("sans_i")
    front["monitor"]["normaliser_after_selector"] = False
    unknown = load_instrument("sans_i")
    unknown["monitor"]["normaliser_after_selector"] = None
    assert load_instrument("sans_i")["monitor"]["normaliser_after_selector"] is True    # AgBe 2025
    for norm, inst, sev, word in (("monitor", behind, "info", "cancels"), ("monitor", front, "check", "in front"),
                                  ("monitor", unknown, "check", "not known"), ("time", behind, "check", "time")):
        r2 = ab.reduce_row_2d(t, t.samples[2], read_tilted, (c.x, c.y), normalise_by=norm, instrument=inst)
        m = [x for x in r2.parameters["missing"] if x["field"] == "selector tilt"]
        assert len(m) == 1 and m[0]["role"] == "water" and m[0]["severity"] == sev and "-4.0°" in m[0]["text"]
        assert word in m[0]["text"]
    r2 = ab.reduce_row_2d(t, t.samples[2], read, (c.x, c.y))
    assert not [x for x in r2.parameters["missing"] if x["field"] == "selector tilt"]


def test_monitor_before_or_after_selector():
    """A monitor in front of the selector keeps its ratio to the proton beam at any λ / tilt."""
    from types import SimpleNamespace as NS
    from pysasfit.sasred.core.monitors import selector_test
    rng = np.random.default_rng(1)
    runs = []
    for lam, tilt, flux in ((6.0, 0.0, 1.0), (6.0, -6.0, 1.6), (10.0, 0.0, 0.35)):
        for _ in range(5):
            p = 1e7 * rng.uniform(0.5, 2)
            m = {"integrated_beam": p, "monitor_front": 0.2 * p * rng.normal(1, 0.01),
                 "monitor1": 0.05 * p * flux * rng.normal(1, 0.01)}
            runs.append(NS(monitors=m, wavelength_A=lam, selector_tilt_deg=tilt, empty_reason=""))
    t = selector_test(runs)
    assert t["monitor_front"].position == "before selector" and t["monitor_front"].after_selector is False
    assert t["monitor1"].position == "after selector" and t["monitor1"].change > 1
    one = selector_test(runs[:5])
    assert one["monitor1"].position == "undetermined" and "two settings" in one["monitor1"].reason


def test_vifor_monitor_test_is_undetermined(vifor):
    """The Vifor folder has 6 Å only: the position of monitor1 cannot be determined from it."""
    from pysasfit.sasred.core.monitors import selector_test
    cat, raws, *_ = vifor
    t = selector_test(list(raws.values()))
    assert t["monitor1"].position == "undetermined"


AGBE = os.environ.get("PYSASRED_TEST_AGBE", "/mnt/user-data/uploads/Ag-Behinate(2025)")


def test_agbe_monitors_and_selector_scans():
    """AgBe 2025 at several λ and two selector speed scans: monitor1 is behind the selector, monitor2 and
    monitor_7 in front; the scans are read (frames summed, kept per frame) and not reduced."""
    import glob
    if not os.path.isdir(AGBE):
        pytest.skip("AgBe 2025 data not available")
    from pysasfit.sasred.core.monitors import selector_test
    from pysasfit.sasred.io.sinq import read_sans_i
    rs = [read_sans_i(f, checksum=False) for f in sorted(glob.glob(os.path.join(AGBE, "*.hdf")))]
    scan = next(r for r in rs if r.run == 1976)
    assert scan.n_frames == 18 and scan.scan.startswith("selector speed scan") and "λ differs" in scan.empty_reason
    assert scan.monitors["monitor1"] == pytest.approx(scan.frames["monitors"]["monitor1"].sum())
    assert scan.frames["wavelength_A"][-1] == pytest.approx(scan.wavelength_A)
    assert scan.counting_time_s == pytest.approx(18 * 600, rel=0.01)
    for sub in (rs, [scan]):                        # all runs, or the scan alone
        t = selector_test(sub)
        assert t["monitor1"].position == "after selector" and t["monitor4"].position == "after selector"
        assert t["monitor2"].position == "before selector" and t["monitor_7"].position == "before selector"
        assert t["monitor_5"].position == "proton beam"


def test_legacy_names_are_read():
    """Files and tables written with L1 / L2 before the rename to coll / SD."""
    from pysasfit.sasred.legacy import rename_config, rename_keys
    from pysasfit.sasred.reduction_table import ReductionTable
    assert rename_keys({"L2_m": 8.0, "L1_m": 8.0, "x": {"L2_effective_m": 8.1}, "L2": 1}) == \
        {"SD_m": 8.0, "coll_m": 8.0, "x": {"SD_effective_m": 8.1}, "SD": 1}
    assert rename_config("L2 8.0 m, λ 6.00 Å, L1 8.0 m") == "SD 8.0 m, λ 6.00 Å, coll 8.0 m"
    t = ReductionTable.from_dict({"config": "L2 8.0 m, λ 6.00 Å, L1 8.0 m",
                                  "efficiency": "geometry:L2 2.0 m, λ 6.00 Å, L1 8.0 m"})
    assert t.config == "SD 8.0 m, λ 6.00 Å, coll 8.0 m" and t.efficiency == "geometry:SD 2.0 m, λ 6.00 Å, coll 8.0 m"
    from pysasfit.toolbox.merge import Curve
    c = Curve(np.ones(2), np.ones(2), np.ones(2), np.zeros(2), "x", "nxcansas", {"L2_m": 2.0, "L1_m": 8.0,
                                                                                 "wavelength_A": 6.0})
    assert c.meta["SD_m"] == 2.0 and "SD 2.00 m" in c.geometry


def test_iq_plot_shows_q_error_bars():
    """With a resolution the I(q) plot has horizontal (q) error bars as well; without, only dI."""
    if not os.path.isdir(RAW):
        pytest.skip("Vifor raw data not available")
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6 import QtWidgets
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    from matplotlib.collections import LineCollection
    from pysasfit.sasred.gui import MainWindow, _qerr
    w = MainWindow()
    w.set_folder(RAW)
    w.select_runs([82130])
    w.update_preview()
    p = w.preview
    if p is None:
        from pysasfit.sasred.reduce import reduce_run_1d
        raw = next(e.raw for e in w.entries if e.run == 82130)
        mask = w._mask(raw.counts.shape)
        p = reduce_run_1d(raw, w.centre_for(raw), w._edges(raw, mask), mask=mask, beam_profile=w.profile_for(raw))
    assert p.q_sigma is not None
    xe = _qerr(p, w.units)
    assert xe.shape == p.q.shape and np.all(xe > 0) and np.all(xe < w.units.q_out(p.q))
    w.preview = p
    w._draw_iq()
    ax = w.iq_canvas.figure.axes[0]
    segs = [c for c in ax.collections if isinstance(c, LineCollection)]
    assert len(segs) >= 2                       # vertical (dI) and horizontal (Qdev) bars
    n_bars = len(segs)
    w.iq_err_q.setChecked(False)                # switched off: only the I error bars
    segs = [c for c in w.iq_canvas.figure.axes[0].collections if isinstance(c, LineCollection)]
    assert len(segs) == n_bars - 1
    w.iq_err_I.setChecked(False)                # both off: points only
    segs = [c for c in w.iq_canvas.figure.axes[0].collections if isinstance(c, LineCollection)]
    assert len(segs) == n_bars - 2
    w.iq_err_q.setChecked(True)
    w.iq_err_I.setChecked(True)
    p.q_sigma = None
    assert _qerr(p, w.units) is None
    w.close()


def test_resolution_preferences(tmp_path, monkeypatch):
    """Apertures and the default dλ/λ come from Settings → Preferences → SANS reduction → Resolution."""
    from types import SimpleNamespace as NS
    from pysasfit.core import settings as prefs
    from pysasfit.sasred.core.beamprofile import collimation_profile
    from pysasfit.sasred.core.resolution import default_dlam
    inst = {"collimation": {"source_aperture_mm": [50, 50], "sample_aperture_mm": [8, 14]}}
    r = NS(coll_m=8.0, SD_m=8.0, SD_effective_m=8.0, wavelength_A=6.0, pixel_size_m=(7.5e-3, 7.5e-3))
    assert default_dlam() == pytest.approx(0.1)
    p0 = collimation_profile(r, inst)
    assert p0.sigma_x_mm == pytest.approx(np.sqrt(50 ** 2 / 12 + 16 ** 2 / 12))
    prefs.set_value("sasred.resolution.sample_aperture_x_mm", 4.0)
    prefs.set_value("sasred.resolution.dlam_default", 0.15)
    try:
        p1 = collimation_profile(r, inst)
        assert p1.sigma_x_mm == pytest.approx(np.sqrt(50 ** 2 / 12 + 8 ** 2 / 12)) and "sample aperture 4×14" in p1.origin()
        assert default_dlam() == pytest.approx(0.15)
        prefs.set_value("sasred.resolution.dlam_default", 0.0)
        assert default_dlam() is None
        inst2 = {"collimation": dict(inst["collimation"], from_preferences=False)}
        assert collimation_profile(r, inst2).sigma_x_mm == pytest.approx(p0.sigma_x_mm)
    finally:
        prefs.set_value("sasred.resolution.sample_aperture_x_mm", 8.0)      # back to the defaults
        prefs.set_value("sasred.resolution.dlam_default", 0.1)


def _beam(SD, a1=(40.0, 45.0), a2=(10.0, 12.0), coll=8.0, runs=(1,), source="empty beam"):
    s = [np.sqrt(((a1[i] * SD / coll) ** 2 + (a2[i] * (coll + SD) / coll) ** 2) / 12) for i in (0, 1)]
    return BeamProfile((SD, 6.0, coll), 64, 64, s[0] / 7.5, s[1] / 7.5, pixel_size_m=(7.5e-3, 7.5e-3), SD_m=SD,
                       wavelength_A=6.0, runs=list(runs), source=source)


def test_divergence_transferred_from_other_distances():
    """No direct beam at this distance: the collimation model calibrated on the direct beams of the same
    collimation (two distances: both apertures; one distance: the guide width), not the divergence copied."""
    from types import SimpleNamespace as NS
    from pysasfit.sasred.core.beamprofile import calibrated_profile, fit_apertures
    a1, a2, how = fit_apertures([_beam(2.0), _beam(8.0)], 8.0, (50, 50), (8, 14))
    assert a1 == pytest.approx((40, 45)) and a2 == pytest.approx((10, 12)) and "fitted" in how
    a1, a2, how = fit_apertures([_beam(8.0, a2=(8, 14))], 8.0, (50, 50), (8, 14))   # one distance
    assert a1 == pytest.approx((40, 45)) and a2 == pytest.approx((8, 14)) and "kept" in how
    a1, a2, _ = fit_apertures([_beam(7.6), _beam(8.0, runs=(1, 2, 3))], 8.0, (50, 50), (10, 12))
    assert a1 == pytest.approx((40, 45))            # distances too close: one-distance calibration
    inst = {"collimation": {"source_aperture_mm": [50, 50], "sample_aperture_mm": [10, 12], "from_preferences": False}}
    r = NS(coll_m=8.0, SD_m=4.0, SD_effective_m=4.0, wavelength_A=6.0, pixel_size_m=(7.5e-3, 7.5e-3))
    p = calibrated_profile(r, {(2.0, 6.0, 8.0): _beam(2.0), (8.0, 6.0, 8.0): _beam(8.0)}, inst)
    exact = _beam(4.0)
    assert p.estimated and p.sigma_x_mm == pytest.approx(exact.sigma_x_mm) and p.sigma_y_mm == pytest.approx(exact.sigma_y_mm)
    assert p.divergence_rad[0] > _beam(8.0).divergence_rad[0]      # shorter distance: wider angular resolution
    assert calibrated_profile(NS(**dict(vars(r), coll_m=4.0)), {(8.0, 6.0, 8.0): _beam(8.0)}, inst) is None
    # a sample transmission run is not used when an empty beam of that collimation exists
    p2 = calibrated_profile(r, {(8.0, 6.0, 8.0): _beam(8.0, a2=(10, 12)),
                                (2.0, 6.0, 8.0): _beam(2.0, a1=(80, 80), source="direct beam (sample transmission runs)")},
                            inst)
    assert "a₁ fitted (direct beam at SD 8 m)" in p2.notes[0]
