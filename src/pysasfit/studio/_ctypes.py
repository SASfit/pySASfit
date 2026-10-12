"""ctypes mirrors of the SASfit C structures (sasfit_function.h, sasfit_plugin.h).

``sasfit_param`` with the layout of ``sasfit_common/include/sasfit_function.h``
(``bool`` is ``int`` there, from ``bool.h``; ``STRLEN`` = 256, ``MAXPAR`` = 50)::

    offset    0  double p[50]
            400  int kernelSelector, kernelSubSelector   (enums)
            408  char errStr[256]
            664  int errLen
            668  int errStatus                            (bool = int)
            672  double *xarr, *yarr
            688  void *moreparam
            696  double more_p[50]
           1096  double (*function)(double, void *)
    size   1104  (64 bit)

SASstudio's original definition used a ``wchar_t *`` for ``errStr``, which
shifted every field after it; ``p[]`` was unaffected.
"""

from __future__ import annotations

import ctypes

MAXPAR = 50
STRLEN = 256
SASFIT_PLUGIN_DEP_ERR = -15


class SasfitParam(ctypes.Structure):
    _fields_ = [
        ("p", ctypes.c_double * MAXPAR),
        ("kernelSelector", ctypes.c_int),
        ("kernelSubSelector", ctypes.c_int),
        ("errStr", ctypes.c_char * STRLEN),
        ("errLen", ctypes.c_int),
        ("errStatus", ctypes.c_int),
        ("xarr", ctypes.POINTER(ctypes.c_double)),
        ("yarr", ctypes.POINTER(ctypes.c_double)),
        ("moreparam", ctypes.c_void_p),
        ("more_p", ctypes.c_double * MAXPAR),
        ("function", ctypes.c_void_p),
    ]

    @classmethod
    def from_values(cls, values=(), more=()):
        s = cls()
        for i, v in enumerate(values):
            s.p[i] = float(v)
        for i, v in enumerate(more):
            s.more_p[i] = float(v)
        return s

    def clear_error(self):
        self.errStatus = 0
        self.errLen = 0
        self.errStr = b""

    @property
    def error(self):
        if not self.errStatus:
            return None
        return self.errStr.decode("latin-1", "replace").strip() or "unspecified SASfit error"


P_PARAM = ctypes.POINTER(SasfitParam)

# scalar sasfit_func_one_t(scalar, sasfit_param *)
FUNC_ONE = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double, P_PARAM)
# scalar sasfit_func_vol_t(scalar, sasfit_param *, int)
FUNC_VOL = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double, P_PARAM, ctypes.c_int)


class PluginFunc(ctypes.Structure):
    """sasfit_plugin_func_t"""

    _fields_ = [
        ("len", ctypes.c_int),
        ("name", ctypes.c_char_p),
        ("func", ctypes.c_void_p),
        ("func_f", ctypes.c_void_p),
        ("func_v", ctypes.c_void_p),
    ]


class PluginInfoHead(ctypes.Structure):
    """sasfit_plugin_info_t without its flexible array member ``functions[]``."""

    _fields_ = [("num", ctypes.c_int), ("functions", PluginFunc * 0)]


def plugin_functions(addr):
    """The ``functions[]`` array of the sasfit_plugin_info_t at address ``addr``."""
    head = PluginInfoHead.from_address(addr)
    off = PluginInfoHead.functions.offset
    return (PluginFunc * head.num).from_address(addr + off)


# int sasfit_plugin_search_t(sasfit_plugin_info_t * requested_set)
SEARCH_T = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.c_void_p)
# int do_init(const sasfit_plugin_info_t **, const sasfit_common_stubs_t *, sasfit_plugin_search_t *)
DO_INIT_ARGS = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, SEARCH_T]
