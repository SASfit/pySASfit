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
# ---- batch 4 (2026-11-23): fuzzysphere, gen_exp_sphere, stochastic_models
_SLD1 = {"default": 1.0, "kind": "sld"}
_SLD0 = {"default": 0.0, "kind": "sld"}
_X = {"lower": 0.0, "upper": 1.0, "kind": "fraction"}
_FZ = {
    "R": {"default": 20.0, "lower": 0.0, **_L},
    "sigma": {"default": 2.0, "lower": 0.0, **_L},
    "eta_sph": _SLD1,
    "eta_sol": _SLD0,
}
_SH = {
    "R_core": {"default": 20.0, "lower": 0.0, **_L},
    "R_tot": {"default": 30.0, "lower": 0.0, **_L},
    "dR": {"default": 10.0, "lower": 0.0, **_L},
    "eta_core": _SLD1,
    "eta_sh": _SLD1,
    "x_in_solv": {"default": 0.0, **_X},
    "x_out_solv": {"default": 1.0, **_X},
    "alpha": {"default": 2.0, "kind": "dimensionless"},
    "eta_solv": _SLD0,
}
_MGD = {
    "W_core": {"default": 15.0, "lower": 0.0, **_L},
    "sigma_core": {"default": 2.0, "lower": 0.0, **_L},
    "W_sh": {"default": 10.0, "lower": 0.0, **_L},
    "sigma_sh,in": {"default": 2.0, "lower": 0.0, **_L},
    "D": {"default": 2.0, "lower": 0.0, **_L},
    "sigma_out": {"default": 3.0, "lower": 0.0, **_L},
    "eta_core": _SLD1,
    "eta_shell": {"default": 0.5, "kind": "sld"},
    "eta_sol": _SLD0,
}
_GES = {
    "Rg": {"default": 10.0, "lower": 0.0, **_L},
    "alpha": {"default": 0.0, "lower": -3.0, "lower_open": True, "kind": "exponent"},
    "Delta eta": _SLD1,
}
_CRW = {
    "scale": {"default": 1.0, "lower": 0.0},
    "phi_1": {
        "default": 0.5,
        "lower": 0.0,
        "upper": 1.0,
        "lower_open": True,
        "upper_open": True,
        "kind": "fraction",
    },
    "d": {"default": 50.0, "lower": 0.0, "lower_open": True, **_L},
    "xi": {"default": 10.0, "lower": 0.0, "lower_open": True, **_L},
    "rc": {"default": 2.0, "lower": 0.0, "lower_open": True, **_L},
}
_BOOL = {
    "Scale": {"default": 1.0, "lower": 0.0},
    "phi_1": _CRW["phi_1"],
    "mu": {"default": 10.0, "lower": 0.0, "lower_open": True, **_L},
    "sigma": {"default": 0.2, "lower": 0.0, "kind": "dimensionless"},
    "dummy": {"default": 0.5, "kind": "dimensionless"},
}
for _n in ("ff_fuzzysphere", "ff_radial_profile_of_fuzzysphere"):
    OVERRIDES[_n] = dict(_FZ)
for _n in ("ff_linshell", "ff_radial_profile_of_linshell", "ff_linshell2", "ff_radial_profile_of_linshell2",
           "ff_expshell", "ff_radial_profile_of_expshell"):  # fmt: skip
    OVERRIDES[_n] = dict(_SH)
for _n in ("ff_coreshellmicrogel", "ff_radial_profile_of_coreshellmicrogel"):
    OVERRIDES[_n] = dict(_MGD)
for _n in ("ff_gen__exp__sphere", "ff_profile__gen__exp__sphere", "ff_gen__gauss__sphere",
           "ff_profile__gen__gauss__sphere"):  # fmt: skip
    OVERRIDES[_n] = dict(_GES)
for _pre in (
    "ff_clipped_random_waves_",
    "ff_clipped_random_waves_twocut_",
    "ff_clipped_random_waves_twocut_intersection_",
):
    for _k in range(1, 6):
        OVERRIDES[f"{_pre}{_k}"] = dict(_CRW)
for _n in ("ff_boolean_union_model", "ff_dead_leaves_model"):
    OVERRIDES[_n] = dict(_BOOL)
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

# magnetic core-shell spheres (ports/magnetic.py, 2026-11-23): SLDs in the unit of the model (e.g. 1e-6 Å⁻²),
# magnetic SLD from pySLDcalculator (2.906e-12 Å⁻² per A/m); pol is replaced by the data-set polarisation setting
_MAG_COMMON = {
    "R": {"default": 5.0, "lower": 0.0, "doc": "core radius"},
    "dR": {"default": 1.0, "lower": 0.0, "doc": "shell thickness"},
    "eta_nuc_core": {"default": 6.9, "doc": "nuclear SLD of the core"},
    "eta_nuc_shell": {"default": 0.5, "doc": "nuclear SLD of the shell"},
    "eta_nuc_matrix": {"default": 6.0, "doc": "nuclear SLD of the matrix"},
    "eta_mag_core": {"default": 1.5, "doc": "magnetic SLD of the core"},
    "psi": {"default": 90.0, "unit": "deg", "kind": "angle",
            "doc": "angle between q and the field on the detector (data-set field direction ψ_H subtracted)"},
    "pol": {"default": 0.0, "lower": -1.0, "upper": 1.0, "kind": "dimensionless",
            "doc": "SASfit polarisation, labelled by the neutron moment: pol = +1 is spin antiparallel to the field; "
                   "not used when the data set has a polarisation setting"},
}
OVERRIDES["ff_magnetic_shell_psi"] = {
    **_MAG_COMMON,
    "eta_mag_shell": {"default": 0.0, "doc": "magnetic SLD of the shell"},
    "eta_mag_matrix": {"default": 0.0, "doc": "magnetic SLD of the matrix"},
}
OVERRIDES["ff_superparamagnetic_ff_psi"] = {
    **_MAG_COMMON,
    "R_av": {"default": 5.0, "lower": 0.0, "lower_open": True, "doc": "radius at which ξ = mu_kT"},
    "mu_kT": {"default": 5.0, "kind": "dimensionless", "doc": "Langevin parameter μ0 μ H / kT of a core of radius R_av"},
}

# batch 5 (2026-11-23): worm-like chains (ports/worms.py), micelles with worm or SAW chains (ports/micelles_worm.py)
_WORM = {
    "lb": {"default": 10.0, "doc": "Kuhn length"},
    "L": {"default": 200.0, "doc": "contour length"},
    "exvol": {"default": 0.0, "lower": 0.0, "kind": "dimensionless",
              "doc": "excluded volume: < 1 without, >= 1 with (method 3: chains with excluded volume)"},
    "dummy": {"default": 0.0, "kind": "dummy"},
    "I0": {"default": 1.0},
}
for _n in ("ff_worms_ps1_", "ff_worms_ps2_", "ff_worms_ps3_", "ff_kholodenko_worm", "ff_freely_joined_chain_of_rods"):
    OVERRIDES[_n] = dict(_WORM)
_MIC5 = {
    "V_brush": {"default": 400.0, "kind": "volume", "doc": "volume of one chain"},
    "V_core": {"default": 400.0, "lower": 0.0, "lower_open": True, "kind": "volume",
               "doc": "volume of one core block"},
    "eta_core": {"default": 6.0},
    "eta_brush": {"default": 1.0},
    "eta_solv": {"default": 0.0},
    "xsolv_core": {"default": 0.0, "lower": 0.0, "upper": 1.0, "upper_open": True, "kind": "dimensionless",
                   "doc": "solvent fraction of the core"},
    "L": {"default": 40.0, "lower": 0.0, "lower_open": True, "kind": "length", "doc": "contour length of a chain"},
    "lb": {"default": 4.0, "lower": 0.0, "kind": "length", "doc": "Kuhn length of a chain"},
}
for _n, _first in (("ff_sphere_chains_saw_", {"R_core": {"default": 5.0, "lower": 0.0},
                                               "n_agg": {"default": 0.02, "lower": 0.0, "kind": "dimensionless",
                                                         "doc": "chains per core surface"}}),
                   ("ff_sphere_chains_saw__rc", {"R_core": {"default": 5.0, "lower": 0.0}}),
                   ("ff_sphere_chains_saw__nagg", {"Nagg": {"default": 50.0, "lower": 0.0, "kind": "count"}})):
    OVERRIDES[_n] = {**_MIC5, **_first}
_RMA5 = {
    **{k: v for k, v in _MIC5.items() if k not in ("L", "lb")},
    "alpha": {"default": 2.0, "lower": 0.0, "kind": "exponent", "doc": "corona density ∝ r^-alpha"},
    "t": {"default": 5.0, "lower": 0.0, "kind": "length", "doc": "corona thickness"},
    "exvol": {"default": 0.0, "lower": 0.0, "kind": "dimensionless",
              "doc": "chains: 0 worm, 1 worm with excluded volume, 2 generalized Gaussian coil (L = Rg, b = nu)"},
}
_LB5 = {"default": 40.0, "lower": 0.0, "lower_open": True, "kind": "length",
        "doc": "contour length L of a chain (exvol 0, 1) or its Rg (exvol 2)"}
_BN5 = {"default": 4.0, "lower": 0.0, "kind": "dimensionless", "doc": "Kuhn length b (exvol 0, 1) or ν (exvol 2)"}
OVERRIDES["ff_sphere_r__a"] = {**_RMA5, "R_core": {"default": 5.0, "lower": 0.0},
                               "n_agg": {"default": 0.02, "lower": 0.0, "kind": "dimensionless"},
                               "L/Rg": _LB5, "b/nu": _BN5}
OVERRIDES["ff_sphere_r__a_rc"] = {**_RMA5, "R_core": {"default": 5.0, "lower": 0.0}, "L/Rg": _LB5, "b/nu": _BN5}
OVERRIDES["ff_sphere_r__a_nagg"] = {**_RMA5, "Nagg": {"default": 50.0, "lower": 0.0, "kind": "count"},
                                    "L (RG)": _LB5, "b (nu)": _BN5}

# batch 6 (2026-11-24): Koyama worm (ports/worms.py), thin_objects cross sections and elongated objects with an
# elliptical core-shell cross-section, thin discs (ports/thin_objects.py)
OVERRIDES["ff_koyama_worm"] = dict(OVERRIDES["ff_kholodenko_worm"])
_SIG6 = {"default": 0.0, "lower": 0.0, "kind": "dimensionless", "doc": "width of the log-normal distribution"}
_CS6 = {
    "R0": {"default": 5.0, "lower": 0.0, "kind": "length", "doc": "core semi-axis R (median)"},
    "R": {"default": 5.0, "lower": 0.0, "kind": "length", "doc": "core semi-axis R (median)"},
    "sigma_R0": _SIG6, "sigma_R": _SIG6, "sigma_t0": _SIG6, "sigma_L0": _SIG6,
    "epsilon": {"default": 1.0, "lower": 0.0, "lower_open": True, "kind": "dimensionless",
                "doc": "ratio of the core semi-axes (εR and R)"},
    "t0": {"default": 2.0, "lower": 0.0, "kind": "length", "doc": "shell thickness (median)"},
    "t": {"default": 2.0, "lower": 0.0, "kind": "length", "doc": "shell thickness"},
    "L0": {"default": 100.0, "lower": 0.0, "kind": "length", "doc": "rod length (median)"},
    "lb": {"default": 20.0, "lower": 0.0, "lower_open": True, "kind": "length", "doc": "Kuhn length"},
    "L": {"default": 500.0, "lower": 0.0, "lower_open": True, "kind": "length", "doc": "contour length"},
    "exvol": {"default": 0.0, "lower": 0.0, "kind": "dimensionless", "doc": "0: without, 1: with excluded volume"},
    "eta_core": {"default": 1.0}, "eta_shell": {"default": 0.5}, "eta_solv": {"default": 0.0},
}
for _n in ("ff_pcs_ellcylsh", "ff_rod_sd_ellcylsh", "ff_worm_ps1_sd_ellcylsh", "ff_worm_ps2_sd_ellcylsh",
           "ff_worm_ps3_sd_ellcylsh", "ff_worm_kholodenko_sd_ellcylsh"):
    OVERRIDES[_n] = dict(_CS6)
OVERRIDES["ff_pcs_homogeneouscyl"] = {"R": {"default": 5.0, "lower": 0.0, "kind": "length"}, "sigma_R": _SIG6,
                                      "eta_core": {"default": 1.0}, "eta_sol": {"default": 0.0}}
OVERRIDES["ff_pcs_homogeneousplate"] = {"t": {"default": 4.0, "lower": 0.0, "kind": "length"}, "sigma_t": _SIG6,
                                        "eta_L": {"default": 1.0}, "eta_solv": {"default": 0.0}}
OVERRIDES["ff_pcs_twoinfinitelythinlayers"] = {"t": {"default": 4.0, "lower": 0.0, "kind": "length"},
                                               "sigma_t": _SIG6}
OVERRIDES["ff_pcs_bilayergauss"] = {
    "sigma_core": {"default": 0.5, "lower": 0.0, "kind": "length", "doc": "width of the Gaussian core layer"},
    "b_core": {"default": -0.5, "kind": "sld", "doc": "scattering length density contrast of the core layer"},
    "sigma_out": {"default": 0.4, "lower": 0.0, "kind": "length", "doc": "width of the Gaussian head layers"},
    "b_out": {"default": 1.0, "kind": "sld", "doc": "scattering length density contrast of the head layers"},
    "D": {"default": 4.0, "lower": 0.0, "kind": "length", "doc": "distance of the head layers"},
    "sigma_D": _SIG6,
}
OVERRIDES["ff_pcs_layeredcentrosymmetricxs"] = {
    "Lc": {"default": 4.0, "lower": 0.0, "kind": "length", "doc": "core layer thickness (median)"},
    "sigma_Lc": _SIG6,
    "Lsh": {"default": 1.0, "lower": 0.0, "kind": "length", "doc": "shell layer thickness (median)"},
    "sigma_Lsh": _SIG6,
    "eta_Lc": {"default": 1.0}, "eta_Lsh": {"default": 0.5}, "eta_solv": {"default": 0.0},
}
OVERRIDES["ff_pcs_plate_chains_rw_"] = {
    "L_core": {"default": 4.0, "lower": 0.0, "kind": "length", "doc": "thickness of the core layer"},
    "n_agg": {"default": 0.05, "lower": 0.0, "lower_open": True, "kind": "dimensionless", "doc": "chains per area"},
    "V_brush": {"default": 100.0, "lower": 0.0, "kind": "volume", "doc": "volume of one chain"},
    "eta_core": {"default": 1.0}, "eta_brush": {"default": 0.4}, "eta_solv": {"default": 0.0},
    "xsolv_core": {"default": 0.0, "lower": 0.0, "upper": 1.0, "upper_open": True, "kind": "dimensionless"},
    "Rg": {"default": 2.0, "lower": 0.0, "lower_open": True, "kind": "length"},
    "d": {"default": 1.0, "lower": 0.0, "kind": "dimensionless", "doc": "distance of the chain centres in units of Rg"},
}
for _n in ("ff_disc_sd_homoxs", "ff_disc_sd_bilayergauss"):
    OVERRIDES[_n] = {"t": {"default": 4.0, "lower": 0.0, "kind": "length"}, "sigma_t": _SIG6,
                     "R0": {"default": 50.0, "lower": 0.0, "kind": "length", "doc": "disc radius (median)"},
                     "sigma_R0": _SIG6, "eta_l": {"default": 1.0}, "eta_sol": {"default": 0.0},
                     "sigma_out": {"default": 0.4, "lower": 0.0, "kind": "length"}, "b_out": {"default": 1.0},
                     "sigma_core": {"default": 0.5, "lower": 0.0, "kind": "length"}, "b_core": {"default": -0.5}}

# batch 7 (2026-11-24): cylinders (ports/cylinders.py), ellipsoidal shells and spheroids (ports/ellipsoidal_shell.py)
_L7 = lambda d, doc: {"default": d, "lower": 0.0, "kind": "length", "doc": doc}  # noqa: E731
_RL7 = {"R": _L7(5.0, "radius"), "L": _L7(40.0, "length"), "eta": {"default": 1.0}}
for _n in ("ff_cylinder", "ff_longcylinder", "ff_porodcylinder"):
    OVERRIDES[_n] = dict(_RL7)
OVERRIDES["ff_longcylinder"] = {"R": _L7(2.0, "radius"), "L": _L7(100.0, "length (L >> R)"), "eta": {"default": 1.0}}
OVERRIDES["ff_flatcylinder"] = {"R": _L7(40.0, "radius"), "L": _L7(2.0, "thickness (L << R, valid for qL < 1)"),
                                "eta": {"default": 1.0, "lower": 0.0}}
OVERRIDES["ff_porodcylinder_v2"] = {"R": _L7(5.0, "radius"),
                                    "nu": {"default": 8.0, "lower": 0.0, "kind": "dimensionless", "doc": "L / R"},
                                    "eta": {"default": 1.0, "lower": 0.0}}
OVERRIDES["ff_disc"] = {"R": _L7(20.0, "radius of the infinitely thin disc"), "eta": {"default": 1.0}}
OVERRIDES["ff_rod"] = {"L": _L7(100.0, "length of the infinitely thin rod"), "eta": {"default": 1.0, "lower": 0.0}}
_CS7 = {"R": _L7(5.0, "core radius (semi-axis R)"), "t": _L7(1.0, "shell thickness"), "L": _L7(40.0, "core length"),
        "epsilon": {"default": 1.0, "lower": 0.0, "lower_open": True, "kind": "dimensionless",
                    "doc": "ratio of the core semi-axes (εR and R)"},
        "eta_core": {"default": 1.0}, "eta_shell": {"default": 0.5}, "eta_solv": {"default": 0.0}}
for _n in ("ff_ellcylshell1", "ff_ellcylshell2", "ff_cylshell1", "ff_cylshell2", "ff_longcylshell"):
    OVERRIDES[_n] = dict(_CS7)
OVERRIDES["ff_longcylshell"]["L"] = _L7(200.0, "length (L >> R + t)")
OVERRIDES["ff_torus"] = {"R": _L7(10.0, "major radius"), "a": _L7(3.0, "tube radius"),
                         "Delta_a": _L7(0.0, "shell thickness of the tube"),
                         "nu": {"default": 1.0, "lower": 0.0, "kind": "dimensionless", "doc": "height / width of the tube"},
                         "eta_c": {"default": 1.0}, "eta_sh": {"default": 0.5}, "eta_sol": {"default": 0.0}}
_ES7 = {"R_polar": _L7(10.0, "polar semi-axis of the core"), "R_equatorial": _L7(5.0, "equatorial semi-axis of the core"),
        "t": _L7(1.0, "shell thickness"),
        "sigma": {"default": 0.0, "lower": 0.0, "kind": "dimensionless",
                  "doc": "width of the log-normal distribution of the scaled lengths"},
        "eta_core": {"default": 1.0}, "eta_shell": {"default": 0.5}, "eta_solv": {"default": 0.0},
        "x_core": {"default": 0.0, "lower": 0.0, "upper": 1.0, "kind": "fraction", "doc": "solvent fraction in the core"},
        "x_shell": {"default": 0.0, "lower": 0.0, "upper": 1.0, "kind": "fraction", "doc": "solvent fraction in the shell"}}
for _n in ("ff_ellipsoidal_shell_0", "ff_ellipsoidal_shell_0t", "ff_ellipsoidal_shell_rp", "ff_ellipsoidal_shell_rp_t",
           "ff_ellipsoidal_shell_re", "ff_ellipsoidal_shell_re_t", "ff_ellipsoidal_shell_2", "ff_ellipsoidal_shell_2t"):
    OVERRIDES[_n] = dict(_ES7)
_SP7 = {"R_polar": _L7(10.0, "polar semi-axis"), "R_equatorial": _L7(5.0, "equatorial semi-axis"),
        "eta_core": {"default": 1.0}, "eta_solv": {"default": 0.0}}
OVERRIDES["ff_spheroid_R"] = dict(_SP7)
OVERRIDES["ff_spheroid_V"] = {**_SP7, "V": {"default": 1000.0, "lower": 0.0, "kind": "dimensionless", "doc": "volume (length³)"}}
OVERRIDES["ff_spheroid_nu"] = {**_SP7, "nu": {"default": 2.0, "lower": 0.0, "kind": "dimensionless",
                                              "doc": "aspect ratio R_polar / R_equatorial"}}
