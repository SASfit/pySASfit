"""Wiring of the pySCAtter parameters: what the GUI offers is what the calculation uses.

For a matrix of settings (particle x interior x orientation x lattice x peak shape x structure factor):

* every parameter the GUI enables (``params.relevant``) changes I(q) or the 2D pattern when it is changed;
* the parameters the GUI greys out do not change them (all of them are changed at once);
* every entry of a combo box gives a result of its own, except the aliases listed in ``ALIASES``.

The default run takes the fast part of the matrix; ``PYSASFIT_WIRING_FULL=1`` adds the slow settings
(oriented cylinders / disks / ellipsoids, cube, triaxial ellipsoid, superball: several minutes).
``python tests/scatter/test_wiring.py [full]`` prints all findings instead of stopping at the first.
"""
import os
import sys
import warnings

import numpy as np
import pytest

from pysasfit.scatter import engine, params as P

FULL = bool(os.environ.get("PYSASFIT_WIRING_FULL"))

BASE = dict(EditRadius=5.0, EditRadiusi=8.0, EditSigma=0.1, Length=20.0, SigmaL=0.1, Alpha=3.0, EditRho=0.5,
            EditDbeta=20.0, theta=60.0, phi=30.0, ucpsi=10.0, ucn1=1.0, ucn2=1.0, ucn3=1.0, uca=30.0, ucb=30.0,
            ucc=30.0, EditDebyeWaller=1.0, EditDomainSize=40.0, EditAzi=8.0, EditPeakPar=2.0, HKLmax=2,
            EditQmax=1.5, EditQmin=0.02, EditQsteps=12, GridPoints=8, I0=2.0, Base=0.01, SigX=30.0, SigY=50.0,
            SigZ=80.0, ShellN=3, ShellThick=2.0, ShellRho2=0.3, EditQmaxPreset=True)
SHOWN = [k for k in P.NUM if k not in P.UNUSED]

#: enabled in the GUI although without effect in this setting: (part of the setting name, key) -> reason
NO_EFFECT_OK = {}
#: combo entries that are the same calculation: combo -> list of sets of indices
ALIASES = {"LType": [{0, 16}, {1, 14}, {2, 15},      # the GISAXS entries: the same lattice in transmission
                     {4, 11},                        # plumber's nightmare: Im3m, the reflections of BCC
                     {12, 13, 19}],                  # CP-Layers, LDQ12: not calculated (disabled in the GUI)
           # mirrored distributions: the same pattern for particles with a centre of symmetry (all of them);
           # Laguerre: calculated as the exponential distribution
           "Ordis": [{0, 8}, {1, 5, 9}, {2, 10}, {3, 11}, {4, 12}]}


def cfg(**kw):
    p = P.defaults()
    p.update(BASE)
    p.update(kw)
    if int(p["ComboBoxParticle"]) == 3 and "Length" not in kw:
        p["Length"] = 9.0                              # vesicle: outer radius
    if int(p["ComboBoxPeak"]) == 4 and "EditPeakPar" not in kw:
        p["EditPeakPar"] = 40.0                        # pseudo-Voigt: eta in percent
    return p


def other(p, k):
    _k, _l, _d, lo, hi, dec, _t = P.NUM[k]
    v = p[k]
    n = v + 1 if dec == 0 else v * 1.25 + 0.05
    if n > hi:
        n = v - 1 if dec == 0 else v * 0.8 - 0.01
    return n


def calc(p):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        q, I1 = engine.calc_1d(p)
        return np.concatenate([q, I1]), engine.calc_2d(p)


def differs(a, b):
    for x, y in zip(a, b):
        if x.shape != y.shape or not np.allclose(x, y, rtol=1e-7, atol=1e-12 * np.nanmax(np.abs(x)),
                                                 equal_nan=True):
            return True
    return False


# ------------------------------------------------------------------------------------------- settings
#: 2D calculations that take seconds: (particle, orientation); part of the full run only
def _slow(p):
    part, od = int(p["ComboBoxParticle"]), int(p["Ordis"])
    if int(p["LType"]) not in (0, 1, 5, 12) and od == 13:
        return True                                    # fibre average of a lattice: 240 rotations
    return part in (4, 6, 8, 9) or (part in (1, 2, 5) and od not in (6, 7))


def settings():
    out = []

    def add(name, **kw):
        out.append((name, cfg(**kw)))
    for part, pn in enumerate(P.PARTICLES):
        if part == 7:
            continue                                   # super ellipsoid: minutes per curve, see test below
        for od in (7, 0, 6, 13):
            if part == 4 and od in (0, 13):
                continue                               # cube: isotropic and z-dir only (CalcError otherwise)
            if part in (3, 9, 10, 11) and od != 7 and not FULL:
                continue                               # no axis: as isotropic (the sphere is in the fast part)
            add(f"{pn} / {P.ORDIS[od]}", ComboBoxParticle=part, Ordis=od)
    for od in range(len(P.ORDIS)):
        if od not in (7, 0, 6, 13):
            add(f"cylinder / {P.ORDIS[od]}", ComboBoxParticle=1, Ordis=od)
    for inter, parts in ((1, (0, 1, 2, 5)), (2, (0,)), (3, (0, 3)), (4, (0, 3))):
        for part in parts:
            add(f"{P.INTERIOR[inter]} / {P.PARTICLES[part]}", ComboBoxInterior=inter, ComboBoxParticle=part)
    add("power-law shell / sphere", ComboBoxInterior=2, ShellProfile=1)
    for lt, kind in P.LTYPE_KIND.items():
        if kind:
            for od in (7, 0, 13):
                add(f"{P.LTYPES[lt]} / sphere / {P.ORDIS[od]}", LType=lt, Ordis=od)
    for pk, name in enumerate(P.PEAKS):
        for od in (7, 0, 13):
            if (pk, od) != (3, 13):                    # mod. 2 Lorentzian: no fibre pattern (CalcError, see below)
                add(f"FCC / {name} / {P.ORDIS[od]}", LType=5, ComboBoxPeak=pk, Ordis=od)
    for ds, para in ((False, True), (False, False)):
        add(f"FCC / width mode DS={ds} para={para}", LType=5, RadButDebyeScherrer=ds, RadioButtonPara=para)
    for lt in (5, 1, 0):
        for od in (7, 0):
            add(f"cylinder on {P.LTYPES[lt]} / {P.ORDIS[od]}", LType=lt, ComboBoxParticle=1, Ordis=od)
    for sf in (1, 2, 3):
        for part in (0, 1) if FULL or sf == 1 else (0,):
            add(f"{P.STRUCTURE_FACTOR_NAMES[sf]} / {P.PARTICLES[part]}", StructureFactor=sf, ComboBoxParticle=part)
    add("detector geometry", EditQmaxPreset=False, EditPixelNoX=8, EditPixelNoY=8, EditDet=2.0, EditPixelX=0.02,
        EditPixelY=0.02)
    return out


SETTINGS = settings()


@pytest.fixture(autouse=True)
def _coarse_table(monkeypatch):
    """A coarse P(q) table for the isotropic 2D patterns: the tests look for changes, not for accuracy."""
    monkeypatch.setattr(engine, "P_TABLE_POINTS", 40)


def findings(name, p):
    """List of wiring faults of one setting."""
    out = []
    ref = calc(p)
    used = P.relevant(p)
    for k in SHOWN:
        if k in used:
            q = dict(p)
            q[k] = other(p, k)
            if not differs(ref, calc(q)) and not any(n in name and kk == k for (n, kk) in NO_EFFECT_OK):
                out.append(f"{name}: '{k}' is enabled but changes nothing")
    q = dict(p)
    for k in SHOWN:
        if k not in used:
            q[k] = other(p, k)
    if differs(ref, calc(q)):                          # find which of the greyed parameters it is
        for k in SHOWN:
            if k not in used:
                q = dict(p)
                q[k] = other(p, k)
                if differs(ref, calc(q)):
                    out.append(f"{name}: '{k}' is greyed out but changes the result")
    return out


@pytest.mark.parametrize("name,p", [pytest.param(n, p, id=n, marks=[] if FULL or not _slow(p) else
                                                 pytest.mark.skip(reason="slow: PYSASFIT_WIRING_FULL=1"))
                                    for n, p in SETTINGS])
def test_enabled_parameters_act_and_greyed_do_not(name, p):
    assert findings(name, p) == []


def test_matrix_covers_every_parameter_and_combo_entry():
    on = set().union(*(P.relevant(p) for _n, p in SETTINGS))
    assert on == set(SHOWN), f"never relevant in the matrix: {set(SHOWN) - on}"
    for key, items in P.COMBOS.items():
        seen = {int(p[key]) for _n, p in SETTINGS}
        missing = set(range(len(items))) - seen - NOT_CALCULATED.get(key, set())
        assert not missing, f"{key}: entries not in the matrix: {[items[i] for i in missing]}"


#: combo entries the engine does not calculate as such (documented; listed so that a change is noticed)
NOT_CALCULATED = {"LType": {13, 19},                   # CP-Layers, LDQ12: treated as 'None'
                  "ComboBoxParticle": {7}}             # super ellipsoid: calculated, but too slow for the matrix


# ------------------------------------------------------------------------------------------- combo boxes
def _combo_results(key, base, entries):
    res = {}
    for i in entries:
        try:
            res[i] = calc(cfg(**dict(base, **{key: i})))
        except engine.CalcError:
            res[i] = None
    return res


def _groups(res):
    """Entries with identical results, as sets of indices."""
    idx = [i for i, r in res.items() if r is not None]
    groups = []
    for i in idx:
        for g in groups:
            if not differs(res[g[0]], res[i]):
                g.append(i)
                break
        else:
            groups.append([i])
    return [set(g) for g in groups if len(g) > 1]


COMBO_CASES = [
    # combo, setting in which it acts, entries, slow
    ("ComboBoxParticle", dict(Ordis=7), [i for i in range(len(P.PARTICLES)) if i != 7], True),
    ("ComboBoxParticle", dict(Ordis=7), [0, 1, 2, 3, 5, 9, 10, 11], False),
    ("ComboBoxInterior", dict(ComboBoxParticle=0), range(len(P.INTERIOR)), False),
    ("ShellProfile", dict(ComboBoxParticle=0, ComboBoxInterior=2), range(len(P.SHELL_PROFILES)), False),
    ("Ordis", dict(ComboBoxParticle=5, theta=60.0, phi=30.0), range(len(P.ORDIS)), True),
    ("Ordis", dict(ComboBoxParticle=5), [6, 7, 13], False),
    ("LType", dict(Ordis=7, ucb=35.0, ucc=40.0), range(len(P.LTYPES)), False),
    ("LType", dict(Ordis=0, ucb=35.0, ucc=40.0), range(len(P.LTYPES)), False),
    ("ComboBoxPeak", dict(LType=5, Ordis=7, EditPeakPar=30.0), range(len(P.PEAKS)), False),
    ("ComboBoxPeak", dict(LType=5, Ordis=0, EditPeakPar=30.0), range(len(P.PEAKS)), False),
    ("StructureFactor", dict(), range(len(P.STRUCTURE_FACTOR_NAMES)), False),
]


@pytest.mark.parametrize("key,base,entries,slow", [
    pytest.param(k, b, list(e), s, id=f"{k} {b}", marks=[] if FULL or not s else
                 pytest.mark.skip(reason="slow: PYSASFIT_WIRING_FULL=1")) for k, b, e, s in COMBO_CASES])
def test_combo_entries_differ(key, base, entries, slow):
    got = _groups(_combo_results(key, base, entries))
    allowed = [a & set(entries) for a in ALIASES.get(key, [])]
    allowed = [a for a in allowed if len(a) > 1]
    names = P.COMBOS[key]
    assert sorted(map(sorted, got)) == sorted(map(sorted, allowed)), \
        f"{key} with {base}: identical results for " + "; ".join(str([names[i] for i in sorted(g)]) for g in got)


def test_width_modes_differ():
    r = [calc(cfg(LType=5, RadButDebyeScherrer=ds, RadioButtonPara=pa))
         for ds, pa in ((True, False), (False, True), (False, False))]
    assert differs(r[0], r[1]) and differs(r[0], r[2]) and differs(r[1], r[2])


def test_orientation_width_acts_for_every_distribution():
    """'particle axis: width' of Gaussian, Exponential, Onsager, Maier-Saupe, Cut-off, Laguerre and the mirrored
    ones: a wider distribution lowers the anisotropy of the pattern of an aligned ellipsoid."""
    def anisotropy(od, width):
        p = cfg(ComboBoxParticle=5, Ordis=od, EditDbeta=width, theta=90.0, phi=0.0, GridPoints=9, EditQmax=0.8)
        I = engine.calc_2d(p) - p["Base"]
        return I[4, 8] / I[8, 4]                       # along qz / along qy at the same |q|
    entries = [0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12] if FULL else [0, 3]
    for od in entries:
        a5, a20, a60 = (anisotropy(od, w) for w in (5.0, 20.0, 60.0))
        assert abs(np.log(a5)) > abs(np.log(a20)) > abs(np.log(a60)), (P.ORDIS[od], a5, a20, a60)
        assert abs(np.log(a5)) > 0.5, P.ORDIS[od]


@pytest.mark.parametrize("kw", [dict(ComboBoxParticle=4, Ordis=0),            # cube: isotropic and z-dir only
                                dict(LType=5, ComboBoxPeak=3, Ordis=13),      # mod. 2 Lorentzian, fibre
                                dict(ComboBoxInterior=1, ComboBoxParticle=4),  # core + shell cube
                                dict(ComboBoxParticle=3, Length=4.0)])         # vesicle: outer < inner radius
def test_unsupported_settings_are_reported(kw):
    """Settings the engine does not calculate end in a CalcError with a message, not in another result."""
    with pytest.raises(engine.CalcError):
        engine.calc_2d(cfg(**kw))


def test_orientation_is_ignored_where_documented():
    """Triaxial ellipsoid: orientation averaged whatever 'orientation' says; the GUI greys the director."""
    a, b = calc(cfg(ComboBoxParticle=6, Ordis=7)), calc(cfg(ComboBoxParticle=6, Ordis=6))
    assert not differs(a, b)
    assert not {"theta", "phi", "EditDbeta"} & P.relevant(cfg(ComboBoxParticle=6, Ordis=0))


if __name__ == "__main__":                            # all findings at once
    engine.P_TABLE_POINTS = 40
    FULL = len(sys.argv) > 1
    for n, p in SETTINGS:
        if FULL or not _slow(p):
            try:
                for f in findings(n, p):
                    print(f, flush=True)
            except Exception as e:                    # noqa: BLE001
                print(f"{n}: ERROR {e!r}", flush=True)
    for k, b, e, s in COMBO_CASES:
        if FULL or not s:
            for g in _groups(_combo_results(k, b, list(e))):
                print(f"{k} with {b}: identical: {[P.COMBOS[k][i] for i in sorted(g)]}", flush=True)
    print("done")
