"""Hand-written defaults and corrections of the generated parameter table (:mod:`.param_table`).

``BY_NAME``: defaults for parameter names that mean the same in every port (unless the port overrides them).
``OVERRIDES``: per port, ``{parameter name or index: {field: value}}`` with the fields of
:class:`.params.Param` (default, lower, upper, lower_open, upper_open, kind, unit, doc).
``RULES``: per port, extra :class:`.params.Rule` conditions between parameters.

Lengths are numbers in the unit reciprocal to q (chosen for q in nm⁻¹).
"""

from __future__ import annotations

from .params import Rule

INF = float("inf")

BY_NAME = {
    "I0": {"default": 1.0},
    "nu": {"default": 0.5},
    "f": {"default": 6.0},
    "kappa": {"default": 2.0, "kind": "dimensionless"},
    "sigma": {
        "default": 0.0,
        "lower": 0.0,
        "kind": "dimensionless",
        "doc": "width of the log-normal size distribution",
    },
    "Wi": {"default": 1.0},
    "theta_0": {"default": 0.0, "unit": "deg"},
    "psi": {"default": 0.0, "unit": "deg"},
    "Delta_theta_0": {"default": 0.0, "unit": "deg"},
    "varphi": {"default": 0.0, "unit": "deg"},
    "eta": {"default": 1.0},
    "eta_sol": {"default": 0.0},
    "eta_solv": {"default": 0.0},
    "eta_s": {"default": 0.0},
}

_SHEARED = {
    0: {"default": 10.0, "lower": 0.0},
    1: {"default": 2.0, "lower": 0.0},
    2: {"default": 100.0, "lower": 0.0},
    "eta_core": {"default": 1.0},
    "eta_shell": {"default": 2.0},
    "psi": {"default": 0.0, "unit": "deg"},
}
_SHEARED_SPH = dict(_SHEARED)
_SHEARED_SPH[2] = {"default": 50.0, "lower": 0.0}
_SHEARFLOW_TWO = {
    "f": {"default": 0.5, "kind": "fraction"},
    "alpha": {"default": 1.0, "kind": "dimensionless"},
    "beta": {"default": 0.5, "kind": "dimensionless"},
    "gamma": {"default": 1.0, "kind": "dimensionless"},
    "delta": {"default": 0.5, "kind": "dimensionless"},
}
_LIN = {
    "fp": {"default": 0.5, "lower": 0.0, "upper": 1.0},
    "alpha": {"default": 1.0, "kind": "dimensionless"},
    "beta": {"default": 0.5, "kind": "dimensionless"},
    "gamma": {"default": 1.0, "kind": "dimensionless"},
    "A": {"default": 1.0, "kind": "dimensionless"},
    "B": {"default": 0.5, "kind": "dimensionless"},
    "C": {"default": 0.0, "kind": "dimensionless"},
    "phi": {"default": 0.0, "unit": "deg"},
    "theta": {"default": 0.0, "unit": "deg"},
    "phi_0": {"default": 0.0, "unit": "deg"},
    "Delta_phi": {"default": 0.0, "unit": "deg", "lower": 0.0},
    "psi_0": {"default": 0.0, "unit": "deg"},
    "P2": {"default": 0.0},
    "P4": {"default": 0.0},
}
_NETW = {
    "Rg": {"default": 10.0},
    "d0": {"default": 5.0, "kind": "length"},
    "lambda": {"default": 1.5, "lower": 0.0, "lower_open": True},
    "gamma": {"default": 1.5},
    "delta": {"default": 0.0, "kind": "angle", "unit": "deg"},
}
_DEFORMED = {"Rg_ll": {"default": 12.0}, "Rg_perp": {"default": 8.0}}
_GAUSS3_UNITS = {
    "b_p": {
        "default": 1e-10,
        "kind": "sld",
        "unit": "cm",
        "doc": "scattering length of the polymer",
    },
    "Mw": {"default": 1e5, "lower": 0.0, "kind": "dimensionless", "unit": "g/mol"},
    "rho_p": {
        "default": 1.0,
        "lower": 0.0,
        "lower_open": True,
        "kind": "dimensionless",
        "unit": "g/cm^3",
    },
    "eta_s": {"default": 0.0, "kind": "sld", "unit": "cm^-2"},
}

OVERRIDES = {
    # ---- spheres
    "ff_spherical_shell_i": {
        "R1": {"default": 10.0},
        "R2": {"default": 8.0},
        "mu": {"default": 0.0, "kind": "dimensionless"},
    },
    "ff_spherical_shell_ii": {
        "nu": {"default": 0.8, "kind": "fraction"},
        "mu": {"default": 0.0, "kind": "dimensionless"},
    },
    "ff_spherical_shell_iii": {
        "R": {"default": 8.0},
        "dR": {"default": 2.0},
        "eta1": {"default": 0.0},
        "eta2": {"default": 1.0},
    },
    "ff_hollow_sphere": {"R": {"default": 8.0}, "dR": {"default": 2.0}},
    "ff_multilamellar_vesicle": {
        "R_c": {"default": 20.0},
        "t_sh": {"default": 4.0},
        "t_sol": {"default": 6.0},
        "eta_shell": {"default": 1.0},
        "n": {"default": 3.0},
    },
    "ff_sphere_with_3_shells": {
        "R_core": {"default": 5.0},
        "t_sh1": {"default": 2.0},
        "t_sh2": {"default": 2.0},
        "t_sh3": {"default": 2.0},
        "eta_core": {"default": 1.0},
        "eta_sh1": {"default": 0.8},
        "eta_sh2": {"default": 0.6},
        "eta_sh3": {"default": 0.4},
    },
    "ff_boucher_sphere": {"alpha": {"default": 4.0, "lower": 2.0, "lower_open": True}},
    "ff_boucher_sphere2": {"alpha": {"default": 4.0}},
    "ff_boucher_profiles": {"alpha": {"default": 4.0}},
    "ff_boucher_profiles2": {"alpha": {"default": 4.0}},
    "ff_kratkysphere": {
        "R": {"default": 5.0},
        1: {"default": 0.05, "kind": "inverse_length"},
        "R0": {"default": 1.0},
        "xi": {"default": 0.0},
        "D": {"default": 2.0, "kind": "exponent"},
        "scale": {"default": 0.0, "kind": "dimensionless"},
    },
    # ---- polymer chains
    "ff_gauss": {"Rg": {"lower": 0.0}},
    "ff_gauss2": {
        "b_p": {"default": 1.0},
        "V": {"default": 0.0, "kind": "dimensionless", "doc": "molecular volume"},
    },
    "ff_gauss3": _GAUSS3_UNITS,
    "ff_gauspoly": {
        "Mw": {"default": 2e4, "unit": "g/mol"},
        "Mn": {"default": 1e4, "lower_open": True, "unit": "g/mol"},
    },
    "ff_generalized_gaussian_coil": {"nu": {"default": 0.588}},
    "ff_generalized_gaussian_coil_2": {
        "nu": {"default": 0.588},
        "b_p": {"default": 1.0},
        "V": {"default": 0.0, "kind": "dimensionless"},
    },
    "ff_generalized_gaussian_coil_3": {"nu": {"default": 0.588}, **_GAUSS3_UNITS},
    # ---- stars, blobs, branched
    "ff_dozierstar": {
        "alpha": {"default": 0.1, "kind": "intensity"},
        "nu": {"default": 0.588},
    },
    "ff_dozierstar2": {"nu": {"default": 0.588}},
    "ff_gauss_exvol_star": {"nu": {"default": 0.588, "lower": 0.0, "lower_open": True}},
    "ff_blob__beaucage_": {"P": {"default": 2.0}},
    "ff_blob__dendrimer_": {"nu": {"default": 0.588}, "xi": {"default": 2.0}},
    "ff_blob__star_": {"nu": {"default": 0.588}},
    "ff_fract_branched_polym": {
        "vm": {"default": 0.588},
        "c": {"default": 1.5, "kind": "exponent"},
    },
    # ---- misc
    "ff_polynom": {
        "p0": {"default": 1.0},
        "p1": {"default": 0.0},
        "p2": {"default": 0.0},
        "p3": {"default": 0.0},
        "p4": {"default": 0.0},
    },
    "ff_cv2": {
        "I_bckgr": {"default": 0.0},
        "x_min": {"default": 0.0},
        "I_scaling": {"default": 1.0},
    },
    "ff_cv": {"m": {"default": 1.0, "kind": "dimensionless"}, "x0": {"default": 0.0}},
    "ff_generalized_form_factor": {
        "sigma_a": {"default": 0.0, "lower": 0.0},
        "alpha": {"default": 1.0},
        "beta": {"default": 1.0},
        "gamma": {"default": 0.0},
        "nu": {"default": 0.5},
    },
    "ff_lesser_generalized_form_factor": {
        "sigma_a": {"default": 0.0, "lower": 0.0},
        "dim": {"default": 3.0},
        "s": {"default": 0.0},
        "p": {"default": 4.0},
    },
    "ff_lesser_generalized_form_factor2": {
        "sigma_a": {"default": 0.0, "lower": 0.0},
        "alpha": {"default": 3.0},
        "s": {"default": 0.0},
        "p": {"default": 4.0},
    },
    # ---- anisotropic polymers
    "ff_deformed_generalized_gaussian_chain": _DEFORMED,
    "ff_deformed_ring_polymer__bmb_": _DEFORMED,
    "ff_deformed_ring_polymer__bz_": _DEFORMED,
    "ff_we__deformed_polym__netw_": _NETW,
    "ff_hsh__deformed_polym__netw_": _NETW,
    "ff_we__deformed_ringpolym__netw_": _NETW,
    "ff_hsh__deformed_ringpolym__netw_": _NETW,
    "ff_reptating_chain": {"lambda": {"default": 1.5}, "t/tau": {"default": 0.1}},
    "ff_polymer_under_shearflow1": {
        "xi": {"default": 1.5, "kind": "exponent", "lower": 1.0, "upper": 2.0},
        "Bx": {"default": 0.5},
        "Bz": {"default": 0.0},
    },
    "ff_ring_under_shearflow1": {
        "xi": {"default": 1.5, "kind": "exponent", "lower": 1.0, "upper": 2.0},
        "Bx": {"default": 0.5},
        "Bz": {"default": 0.0},
    },
    "ff_polymer_under_shearflow2": _SHEARFLOW_TWO,
    "ff_ring_under_shearflow2": _SHEARFLOW_TWO,
    "ff_shearflow_lin": _LIN,
    "ff_shearflow_lin_g1": _LIN,
    "ff_shearflow_lin_g2": _LIN,
    "ff_sheared_cylinders__legendre_": {
        "P2": {"default": 0.4},
        "P4": {"default": 0.1},
        "P6": {"default": 0.0},
    },
}
for _kind in (
    "maier_saupe",
    "gauss",
    "boltzmann",
    "onsager",
    "heavyside",
    "hayterpenfold",
    "legendre",
):
    OVERRIDES.setdefault(f"ff_sheared_cylinders__{_kind}_", {}).update(_SHEARED)
    if _kind != "legendre":
        OVERRIDES[f"ff_sheared_spheroids__{_kind}_"] = dict(_SHEARED_SPH)
    _SZ = {"sigma": {"default": 0.0, "lower": 0.0, "upper": 1.0, "upper_open": True, "kind": "dimensionless",
                     "doc": "relative standard deviation of the Schulz-Zimm size distribution"}}  # fmt: skip
    OVERRIDES.setdefault(f"ff_sheared_cylinders__{_kind}__schulz_", {}).update(
        {**OVERRIDES[f"ff_sheared_cylinders__{_kind}_"], **_SZ}
    )
    if _kind != "legendre":
        OVERRIDES[f"ff_sheared_spheroids__{_kind}__schulz_"] = {**_SHEARED_SPH, **_SZ}

# ---- batch 2 (2026-11-18): non_particulate, linear_pearls, pearl_necklace, spherical_ellipsoidal_micelles
_I = {"kind": "intensity"}
_LEN = {"kind": "length"}
_EXP = {"kind": "exponent"}
OVERRIDES.update(
    {
        "ff_beaucage_exppowlaw": {
            "G": {"default": 1.0, **_I},
            "B": {"default": 1e-3, **_I},
            "Gs": {"default": 0.1, **_I},
            "Bs": {"default": 1e-3, **_I},
            "Rg": {"default": 10.0, **_LEN},
            "Rsub": {"default": 1.0, **_LEN},
            "Rs": {"default": 1.0, **_LEN},
            "P": {"default": 4.0, **_EXP},
            "Ps": {"default": 4.0, **_EXP},
        },
        "ff_beaucage_exppowlaw_2": {
            "G": {"default": 1.0, **_I},
            "B": {"default": 1e-3, **_I},
            "Rg_i": {"default": 10.0, **_LEN},
            "Rg_i+1": {"default": 1.0, **_LEN},
            "k": {"default": 1.0, "kind": "dimensionless"},
            "P": {"default": 4.0, **_EXP},
        },
        "ff_broad_peak": {
            "xi": {"default": 10.0},
            "q0": {"default": 0.3, "kind": "inverse_length"},
            "m": {"default": 2.0},
            "p": {"default": 1.0},
        },
        "ff_debye_anderson_brumberger": {"xi": {"default": 5.0}},
        "ff_g_dab": {"xi": {"default": 5.0}, "H": {"default": 0.5, "kind": "exponent"}},
        "ff_def_g_dab": {
            "xi": {"default": 5.0},
            "H": {"default": 0.5, "kind": "exponent"},
            "epsilon": {"default": 1.5},
        },
        "ff_extended_guinier_law": {
            "a": {"default": 1.0, "kind": "exponent"},
            "Ra": {"default": 2.0, **_LEN},
        },
        "ff_generalized_guinier_porod_law": {
            "G2": {"default": 1.0, **_I},
            "s2": {"default": 0.5, "upper": 3.0, "upper_open": True, **_EXP},
            "Rg2": {"default": 20.0},
            "s1": {"default": 1.5, "upper": 3.0, "upper_open": True, **_EXP},
            "Rg1": {"default": 3.0},
            "m": {"default": 4.0},
        },
        "ff_ornstein_zernike": {"xi": {"default": 5.0}},
        "ff_spinodal": {
            "Imax": {"default": 1.0, **_I},
            "Qmax": {"default": 0.3, "kind": "inverse_length"},
            "gamma": {"default": 2.0, "kind": "dimensionless"},
        },
        "ff_spinodal2": {
            "Imax": {"default": 1.0, **_I},
            "Qmax": {"default": 0.3, "kind": "inverse_length"},
            "phi/gamma": {"default": 0.8, "kind": "dimensionless"},
            "d": {"default": 0.1, "kind": "dimensionless"},
            4: {
                "default": 0.0,
                "kind": "count",
                "lower": 0.0,
                "upper": 3.0,
                "doc": "0: gamma given, 1-3: gamma(phi) of 3D/2D/1D models",
            },
        },
        "ff_teubner_strey": {"xi": {"default": 10.0}, "d": {"default": 20.0}},
        "ff_linear_pearls": {
            "R": {"default": 5.0},
            "edge_sep": {"default": 2.0, **_LEN},
            "num_pearls": {"default": 5.0, "kind": "count"},
        },
    }
)
_PN_DEF = {
    "R": {"default": 5.0},
    "l": {"default": 10.0},
    "beta_r": {"default": 1.0, "kind": "dimensionless"},
    "beta_s": {"default": 1.0, "kind": "dimensionless"},
    "Np": {"default": 5.0},
}
for _n in ("ff_pearl_necklace", "ff_pearl_necklace_ext", "ff_pearl_necklace_mod"):
    OVERRIDES[_n] = dict(_PN_DEF)
OVERRIDES["ff_pearl_necklace_ext"]["g"] = {"default": 1.0}
_MIC = {
    "V_brush": {
        "default": 5.0,
        "doc": "molecular volume of a chain (length unit cubed)",
    },
    "V_core": {
        "default": 5.0,
        "doc": "molecular volume of the core block (length unit cubed)",
    },
    "eta_core": {"default": 1.0},
    "eta_brush": {"default": 0.5},
    "eta_solv": {"default": 0.0},
    "xsolv_core": {"default": 0.0},
    "Rg": {"default": 2.0},
    "d": {"default": 1.0, "kind": "dimensionless"},
    "R_core": {"default": 5.0},
    "n_agg": {"default": 0.3, "doc": "chains per surface area"},
    "Nagg": {"default": 50.0, "kind": "count"},
    "epsilon": {"default": 1.5, "lower": 0.0, "lower_open": True},
}
for _n in (
    "ff_sphere_chains_rw_",
    "ff_sphere_chains_rw__rc",
    "ff_sphere_chains_rw__nagg",
    "ff_ell_chains_rw_",
    "ff_ell_chains_rw__rc",
    "ff_ell_chains_rw__nagg",
):
    OVERRIDES[_n] = dict(_MIC)
OVERRIDES["ff_sphere_chains_rw__rc"]["V_core"] = {
    **_MIC["V_core"],
    "lower": 0.0,
    "lower_open": True,
}

# ---- batch 3 (2026-11-18): fractal_obj, fractal_series, powerlaw, guwelop
_L = {"kind": "length"}
_FR = {
    "Rg": {"default": 10.0, "lower": 0.0},
    "D": {"default": 2.0, "kind": "exponent"},
    "I0": {"default": 1.0, **_I},
}
for _n in (
    "ff_fisher_burford",
    "ff_massfractexp",
    "ff_massfractgauss",
    "ff_dlcaggregation",
    "ff_rlcaggregation",
    "ff_aggregate__overlapsph_cut_off_",
):
    OVERRIDES[_n] = dict(_FR)
OVERRIDES["ff_aggregate__overlapsph_cut_off_"]["D"] = {
    "default": 2.0,
    "kind": "exponent",
    "lower": 0.0,
    "lower_open": True,
}
OVERRIDES["ff_aggregate__exp__x_a__cut_off_"] = {
    "xi": {"default": 10.0},
    "D": {"default": 2.0, "kind": "exponent"},
    "I0": {"default": 1.0, **_I},
    "alpha": {"default": 1.0, "kind": "exponent", "lower": 0.0, "lower_open": True},
}
OVERRIDES["ff_dumbbellshell"] = {
    "R": {"default": 5.0, "lower": 0.0},
    "dR": {"default": 1.0, "lower": 0.0},
    "L": {"default": 2.0},
    "eta_c": {"default": 1.0},
    "eta_sh": {"default": 2.0},
}
_TAS = {
    "R1": {"default": 5.0},
    "R2": {"default": 3.0},
    "eta_0": {"default": 0.0},
    "eta_1": {"default": 1.0},
    "eta_2": {"default": 2.0},
}
OVERRIDES["ff_two_attached_spheres"] = dict(_TAS)
OVERRIDES["ff_two_attached_spheres_w_distr"] = {
    **_TAS,
    "sigma1": {"default": 0.1, "kind": "dimensionless"},
    "sigma2": {"default": 0.1, "kind": "dimensionless"},
}
_DSC = {
    "R_c": {"default": 5.0, "lower": 0.0, **_L},
    "DR1": {"default": 1.0, "lower": 0.0, **_L},
    "DR2": {"default": 1.0, "lower": 0.0, **_L},
    "L": {"default": 2.0, **_L},
    "eta_c": {"default": 1.0},
    "eta_1": {"default": 2.0},
    "eta_2": {"default": 1.5},
    "eta_sol": {"default": 0.0},
    "x_sol_sh2": {"default": 0.0, "lower": 0.0, "upper": 1.0, "kind": "fraction"},
    "n": {"default": 3.0},
}
OVERRIDES["ff_doubleshellchain"] = dict(_DSC)
OVERRIDES["ff_tetrahedrondoubleshell"] = {**_DSC, "n": {"default": 4.0}}
OVERRIDES["ff_stackdiscs"] = {
    "R": {"default": 20.0},
    "t": {"default": 2.0},
    "h": {"default": 1.0},
    "n": {"default": 5.0},
    "eta_c": {"default": 1.0},
    "eta_l": {"default": 2.0},
    "sigma": {"default": 1.0, "lower": 0.0, "lower_open": True, **_L},
    "D": {"default": 8.0},
}
OVERRIDES["ff_fractal_series1"] = {
    "N": {"default": 1.0},
    "Rg_min": {"default": 1.0},
    "fd1": {"default": 2.0, **_EXP},
    "Rg_max": {"default": 100.0},
}
OVERRIDES["ff_fractal_series2"] = {
    "N": {"default": 1.0},
    "xmin": {"default": 1.0, **_L},
    "fd1": {"default": 2.0, **_EXP},
    "x12": {"default": 10.0, **_L},
    "fd2": {"default": 2.5, **_EXP},
    "xmax": {"default": 100.0, **_L},
}
OVERRIDES["ff_fractal_series3"] = {
    "N": {"default": 1.0, "lower": 0.0},
    "xmin": {"default": 1.0, **_L},
    "fd1": {"default": 2.0, **_EXP},
    "x12": {"default": 5.0, **_L},
    "fd2": {"default": 2.5, **_EXP},
    "x23": {"default": 20.0, **_L},
    "fd3": {"default": 2.0, **_EXP},
    "xmax": {"default": 100.0, **_L},
}
_PL = {
    "R": {"default": 10.0, "lower": 0.0},
    "alpha": {"default": 2.0, "lower": -3.0, "lower_open": True},
    "Deta": {"default": 1.0, "kind": "sld"},
}
OVERRIDES["ff_power_law"] = dict(_PL)
OVERRIDES["ff_profile__power_law"] = dict(_PL)
_GW = {
    "eta": {"default": 1.0},
    "bu": {"default": 50.0, **_L},
    "theta": {"default": 1.0, "kind": "dimensionless"},
    "alpha": {"default": 2.0, "lower": 0.0, "lower_open": True},
    "beta": {"default": 1.0, "lower": 0.0, "lower_open": True},
    "gamma": {"default": 1.0, "lower": 0.0, "lower_open": True},
    "lambda": {"default": 5.0, "lower": 0.0, "lower_open": True, **_L},
}
for _n in (
    "ff_guwelop_sphere",
    "ff_guwelop_cyl_Pcs",
    "ff_guwelop_planar_Pcs",
    "ff_guwelop_profile",
):
    OVERRIDES[_n] = dict(_GW)
# ---- ribbon (2026-11-18)
_RB = {
    "A": {"default": 2.0, "lower": 0.0, **_L},
    "sigma_a": {"default": 0.5, "lower": 0.0, "lower_open": True, **_L},
    "B": {"default": 20.0, "lower": 0.0, **_L},
    "sigma_B": {"default": 4.0, "lower": 0.0, **_L},
    "C": {"default": 200.0, "lower": 0.0, **_L},
    "eta": {"default": 1.0, "kind": "sld"},
}
OVERRIDES["ff_ribbon_mono"] = {k: v for k, v in _RB.items() if not k.startswith("sigma")}
OVERRIDES["peaks_ribbon_w_sd"] = dict(_RB)
OVERRIDES["peaks_ribbon_w_sd"]["A"] = {"default": 2.0, "lower": 0.0, "lower_open": True, **_L}
RULES_B3 = {
    "ff_guwelop_sphere": [Rule(lambda p: p[4] != 0, "theta == 0")],
    "ff_guwelop_cyl_Pcs": [Rule(lambda p: p[4] != 0, "theta == 0")],
    "ff_guwelop_planar_Pcs": [Rule(lambda p: p[4] != 0, "theta == 0")],
    "ff_guwelop_profile": [Rule(lambda p: p[4] != 0, "theta == 0")],
}

RULES = {
    "ff_gauspoly": [Rule(lambda p: p[1] >= p[2], "Mw < Mn")],
    "ff_generalized_guinier_porod_law": [
        Rule(
            lambda p: (
                p[1] == p[3] or p[2] ** 2 * (3.0 - p[3]) > p[4] ** 2 * (3.0 - p[1])
            ),
            "Rg2^2/(3-s2) <= Rg1^2/(3-s1) (needed for s2 < s1)",
        ),
    ],
}
RULES.update(RULES_B3)
