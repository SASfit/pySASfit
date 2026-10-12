"""Loading libsasfit and the SASfit plugin libraries.

SASfit plugins are not linked against libsasfit. They call the common SASfit
functions (integration, special functions, error handling, ...) through a
table of function pointers handed to them in ``do_init()``; before that every
such call dereferences a NULL pointer. The loader therefore does what the
SASfit core (``sasfit_plugin_backend.c``) does:

1. load ``libsasfit`` and fetch the table with ``sasfit_common_stubs_ptr()``;
2. for each plugin library call ``do_init(&exported, stubs, search)``, where
   ``search`` resolves functions the plugin imports from other plugins (here a
   Python callback over the functions already loaded);
3. register the exported ``sasfit_plugin_func_t`` entries (function, ``_f``
   and ``_v`` pointers) by name.

A plugin whose imports are not loaded yet returns ``SASFIT_PLUGIN_DEP_ERR``;
then all plugin libraries are loaded and the plugin is initialised again.

The integration/precision settings of libsasfit (``sasfit_eps``) are zero
until set; :data:`DEFAULT_PRECISION` (the SASfit GUI defaults) are applied on
loading and can be changed with :meth:`SASfitLibrary.set_precision`.
"""

from __future__ import annotations

import ctypes
import os
import sys
import threading

from . import paths
from ._ctypes import DO_INIT_ARGS, SASFIT_PLUGIN_DEP_ERR, SEARCH_T, plugin_functions

__all__ = ["SASfitLibrary", "SASfitLibraryNotFound", "get_library", "DEFAULT_PRECISION", "QUADRATURE"]


class SASfitLibraryNotFound(RuntimeError):
    pass


QUADRATURE = {
    "OOURA_DOUBLE_EXP_QUADRATURE": 0, "OOURA_CLENSHAW_CURTIS_QUADRATURE": 1, "GSL_CQUAD": 2, "GSL_QAG": 3,
    "H_CUBATURE": 4, "P_CUBATURE": 5, "NR_QROMB": 6, "TanhSinh": 8, "GSL_QAGS": 9, "GSL_QNG": 10,
    "GSL_G962D": 11, "GSL_GAUSSLEGENDRE": 12, "GSL_CHEBYSHEV1": 13, "GSL_CHEBYSHEV2": 14, "GSL_GEGENBAUER": 15,
    "GSL_EXPONENTIAL": 16, "GSL_LAGUERRE": 17, "GSL_JACOBI": 18, "GSL_MC_MISER": 19, "GSL_MC_VEGAS": 20,
    "GSL_MC_PLAIN": 21, "TANHSINH_1": 22, "TANHSINH_2": 23,
}  # fmt: skip
SPHAVG = {
    "SPHAVG_GSL_2D_GAUSSLEGENDRE": 0, "SPHAVG_Lebedev": 1, "SPHAVG_FIBONACCI": 2, "SPHAVG_HCUBATURE": 3,
    "SPHAVG_PCUBATURE": 4, "SPHAVG_MC_MISER": 5, "SPHAVG_MC_VEGAS": 6, "SPHAVG_MC_PLAIN": 7,
    "SPHAVG_SPHERICAL_T_DESIGN": 8,
}  # fmt: skip
HANKEL = {"HANKEL_OOURA_DEO": 0, "HANKEL_OGATA_2005": 1, "HANKEL_FBT0": 2, "HANKEL_QWE": 12}
ROOTALG = {"BISECTION": 0, "FALSE_POSITION": 1, "BRENT": 2, "NEWTON": 3, "SECANT": 4, "STEFFENSON": 5}

# name: (setter, getter, ctype, SASfit default) -- defaults from SASstudio dev/sasfit_function_dict.ipynb
_EPS = {
    "nriq": ("sasfit_eps_set_nriq", "sasfit_eps_get_nriq", ctypes.c_double, 1e-10),
    "aniso": ("sasfit_eps_set_aniso", "sasfit_eps_get_aniso", ctypes.c_double, 1e-5),
    "res": ("sasfit_eps_set_res", "sasfit_eps_get_res", ctypes.c_double, 1e-5),
    "h": ("sasfit_eps_set_h", "sasfit_eps_get_h", ctypes.c_double, 1e-5),
    "jmax_nriq": ("sasfit_eps_set_jmax_nriq", "sasfit_eps_get_jmax_nriq", ctypes.c_int, 18),
    "jmax_aniso": ("sasfit_eps_set_jmax_aniso", "sasfit_eps_get_jmax_aniso", ctypes.c_int, 18),
    "jmax_res": ("sasfit_eps_set_jmax_res", "sasfit_eps_get_jmax_res", ctypes.c_int, 18),
    "int_strategy": ("sasfit_set_int_strategy", "sasfit_get_int_strategy", ctypes.c_int, QUADRATURE["P_CUBATURE"]),
    "sphavg_strategy": ("sasfit_set_sphavg_strategy", "sasfit_get_sphavg_strategy", ctypes.c_int, SPHAVG["SPHAVG_FIBONACCI"]),
    "hankel_strategy": ("sasfit_set_hankel_strategy", "sasfit_get_hankel_strategy", ctypes.c_int, HANKEL["HANKEL_OOURA_DEO"]),
    "sincos_quad_strategy": ("sasfit_set_sincos_quad_strategy", "sasfit_get_sincos_quad_strategy", ctypes.c_int, 0),
    "gausslegendre": ("sasfit_eps_set_gausslegendre", "sasfit_eps_get_gausslegendre", ctypes.c_int, 128),
    "fibonacci": ("sasfit_eps_set_fibonacci", "sasfit_eps_get_fibonacci", ctypes.c_int, 1014),
    "lebedev": ("sasfit_eps_set_lebedev", "sasfit_eps_get_lebedev", ctypes.c_int, 32),
    "spherical_t_design": ("sasfit_eps_set_spherical_t_design", "sasfit_eps_get_spherical_t_design", ctypes.c_int, 42),
    "iter_4_mc": ("sasfit_eps_set_iter_4_mc", "sasfit_eps_get_iter_4_mc", ctypes.c_int, 1001),
    "sg_level": ("sasfit_set_sg_level", "sasfit_get_sg_level", ctypes.c_int, 8),
    "rootalg": ("sasfit_set_rootalg", "sasfit_get_rootalg", ctypes.c_int, ROOTALG["BRENT"]),
}  # fmt: skip
DEFAULT_PRECISION = {k: v[3] for k, v in _EPS.items()}


class SASfitLibrary:
    """libsasfit plus the initialised plugin functions (use :func:`get_library`)."""

    def __init__(self, lib_dir=None, precision=None):
        lib_dir = os.path.abspath(lib_dir) if lib_dir else paths.lib_dir()
        core = os.path.join(lib_dir, f"libsasfit.{paths.LIB_EXT}") if lib_dir else None
        if not core or not os.path.isfile(core):
            raise SASfitLibraryNotFound(
                "SASfit libraries not found. Install them once with\n"
                "    pysasfit-studio install-binaries <SASstudio or SASfit folder>\n"
                "or set PYSASFIT_SASFIT_LIB to a folder containing "
                f"libsasfit.{paths.LIB_EXT} (and plugins/libsasfit_*.{paths.LIB_EXT})."
            )
        self.lib_dir = lib_dir
        self.plugin_dirs = [d for d in (os.path.join(lib_dir, "plugins"), lib_dir) if os.path.isdir(d)]
        self._dll_dir_handles = []
        if sys.platform == "win32":  # dependencies (gsl, fftw, mingw runtime) next to libsasfit
            for d in [lib_dir] + self.plugin_dirs:
                self._dll_dir_handles.append(os.add_dll_directory(d))
        self.core = ctypes.CDLL(core)
        self.core.sasfit_common_stubs_ptr.restype = ctypes.c_void_p
        self.stubs = self.core.sasfit_common_stubs_ptr()
        self.registry = {}  # name -> (func, func_f, func_v) addresses
        self.loaded = {}  # library path -> CDLL
        self.failed = {}  # library path -> reason
        self._search_cb = SEARCH_T(self._search)  # keep a reference
        self._lock = threading.RLock()
        p = dict(DEFAULT_PRECISION)
        p.update(precision or {})
        self.set_precision(**p)

    # ------------------------------------------------------------ precision
    def set_precision(self, **values):
        """Set libsasfit integration settings, e.g. ``set_precision(nriq=1e-8, int_strategy='GSL_QAGS')``."""
        for k, v in values.items():
            if k not in _EPS:
                raise KeyError(f"unknown setting '{k}'; known: {', '.join(_EPS)}")
            setter, _, ct, _ = _EPS[k]
            if isinstance(v, str):
                v = {**QUADRATURE, **SPHAVG, **HANKEL, **ROOTALG}[v]
            f = getattr(self.core, setter, None)
            if f is None:
                continue  # older libsasfit without this setting
            f.argtypes = [ct]
            f.restype = None
            f(ct(v))

    def get_precision(self):
        out = {}
        for k, (_, getter, ct, _) in _EPS.items():
            f = getattr(self.core, getter, None)
            if f is not None:
                f.restype = ct
                out[k] = f()
        return out

    # --------------------------------------------------------------- plugins
    def _search(self, req_addr):
        """sasfit_plugin_search: fill the requested imports from the registry."""
        if not req_addr:
            return 0
        for rec in plugin_functions(req_addr):
            name = ctypes.string_at(rec.name, rec.len).decode()
            hit = self.registry.get(name)
            if hit is None or not hit[0]:
                return SASFIT_PLUGIN_DEP_ERR
            rec.func, rec.func_f, rec.func_v = hit
        return 1

    def plugin_files(self):
        out = []
        for d in self.plugin_dirs:
            out += [
                os.path.join(d, f)
                for f in sorted(os.listdir(d))
                if f.startswith("libsasfit_") and f.endswith("." + paths.LIB_EXT)
            ]
        return out

    def find_plugin_file(self, collection):
        for d in self.plugin_dirs:
            f = os.path.join(d, f"lib{collection}.{paths.LIB_EXT}")
            if os.path.isfile(f):
                return f
        return None

    def load_plugin(self, filename):
        """Initialise one plugin library; returns True, or False on a missing dependency."""
        with self._lock:
            filename = os.path.abspath(filename)
            if filename in self.loaded:
                return True
            dll = ctypes.CDLL(filename)
            dll.get_ver.restype = ctypes.c_int
            dll.do_init.argtypes = DO_INIT_ARGS
            dll.do_init.restype = ctypes.c_int
            exp = ctypes.c_void_p()
            res = dll.do_init(ctypes.byref(exp), self.stubs, self._search_cb)
            if res == SASFIT_PLUGIN_DEP_ERR:
                return False
            if res == 0 or not exp.value:
                self.failed[filename] = f"do_init returned {res}"
                raise RuntimeError(f"could not initialise {filename} (do_init returned {res})")
            for rec in plugin_functions(exp.value):
                name = ctypes.string_at(rec.name, rec.len).decode()
                self.registry[name] = (rec.func, rec.func_f, rec.func_v)
            self.loaded[filename] = dll
            return True

    def load_all(self):
        """Load every plugin library, resolving inter-plugin dependencies by repeated passes."""
        with self._lock:
            pending = [f for f in self.plugin_files() if f not in self.loaded]
            while pending:
                still = []
                for f in pending:
                    try:
                        if not self.load_plugin(f):
                            still.append(f)
                    except (OSError, RuntimeError) as e:
                        self.failed[f] = str(e)
                if len(still) == len(pending):
                    for f in still:
                        self.failed[f] = "unresolved dependency"
                    break
                pending = still
            return len(self.loaded)

    def core_pointers(self, name):
        """(func, func_f, func_v) addresses of a function built into libsasfit (None if absent)."""

        def addr(sym):
            try:
                return ctypes.cast(getattr(self.core, sym), ctypes.c_void_p).value
            except AttributeError:
                return None

        f = addr("sasfit_" + name)
        if f is None:
            raise KeyError(f"libsasfit exports no function sasfit_{name}")
        return (f, addr(f"sasfit_{name}_f"), addr(f"sasfit_{name}_v"))

    def pointers(self, name, collection=None):
        """(func, func_f, func_v) addresses of a model function, loading its library if needed."""
        if collection == "libsasfit":
            return self.core_pointers(name)
        with self._lock:
            if name not in self.registry:
                f = self.find_plugin_file(collection) if collection else None
                if f is None or not self.load_plugin(f):
                    self.load_all()
                    if f is not None:
                        self.load_plugin(f)
            if name not in self.registry:
                try:  # some functions exist both as plugin and in libsasfit
                    return self.core_pointers(name)
                except KeyError:
                    pass
                raise KeyError(f"SASfit function '{name}' not exported by any loaded plugin library")
            return self.registry[name]


_LIB = None
_LIB_LOCK = threading.Lock()


def get_library(lib_dir=None):
    """The process-wide :class:`SASfitLibrary` (created on first use)."""
    global _LIB
    with _LIB_LOCK:
        if _LIB is None or (lib_dir and os.path.abspath(lib_dir) != _LIB.lib_dir):
            _LIB = SASfitLibrary(lib_dir)
        return _LIB
