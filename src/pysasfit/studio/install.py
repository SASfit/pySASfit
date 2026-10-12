"""Copy the compiled SASfit libraries into pysasfit (run once per installation).

The libraries are platform specific and large (~90 MB on Windows), so they are
not kept in the git repository. ``install_binaries`` copies them from a
SASstudio checkout or an unpacked SASfit release:

* ``libsasfit.<ext>`` and the runtime libraries next to it (gsl, gslcblas,
  fftw, zlib, the MinGW runtime on Windows) -> ``<dest>/``
* ``libsasfit_*.<ext>`` plugin libraries and their ``sasfit_*.h`` headers
  -> ``<dest>/plugins/`` (from ``<source>/plugins`` or, for a SASfit release,
  ``<source>/plugins/<collection>/lib`` and ``.../include``)
* optionally the ``FrolovPoints`` cubature tables (~100 MB).

``dest`` defaults to the ``sasfit_lib`` folder inside the package, which
:mod:`pysasfit.studio.paths` searches automatically.
"""

from __future__ import annotations

import os
import shutil

from . import paths

__all__ = ["install_binaries"]

_RUNTIME_PREFIXES = ("libsasfit", "libgsl", "libgslcblas", "libfftw", "zlib", "libgcc", "libstdc++", "libwinpthread",
                     "libgomp", "libquadmath", "libgfortran")  # fmt: skip


def _copy(src, dst, log):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst) and os.path.getmtime(dst) >= os.path.getmtime(src) and os.path.getsize(dst) == os.path.getsize(src):
        return 0
    shutil.copy2(src, dst)
    log(f"  {os.path.relpath(dst, os.path.dirname(os.path.dirname(dst)))}")
    return 1


def install_binaries(source, dest=None, frolov=False, verbose=True):
    """Copy SASfit libraries from ``source`` to ``dest``; returns the number of files copied."""
    ext = "." + paths.LIB_EXT
    dest = os.path.abspath(dest or paths.DEFAULT_LIB)
    source = os.path.abspath(source)
    log = print if verbose else (lambda *a: None)
    if not os.path.isdir(source):
        raise FileNotFoundError(source)
    n = 0
    log(f"copying SASfit libraries from {source} to {dest}")
    for f in os.listdir(source):  # core + runtime
        if f.endswith(ext) and f.startswith(_RUNTIME_PREFIXES) and not f.startswith("libsasfit_"):
            n += _copy(os.path.join(source, f), os.path.join(dest, f), log)
    plug = os.path.join(source, "plugins")
    if os.path.isdir(plug):
        for root, _, files in os.walk(plug):
            for f in files:
                if (f.startswith("libsasfit_") and f.endswith(ext)) or (f.startswith("sasfit_") and f.endswith(".h")):
                    n += _copy(os.path.join(root, f), os.path.join(dest, "plugins", f), log)
                elif frolov and "FrolovPoints" in root:
                    n += _copy(os.path.join(root, f), os.path.join(dest, "plugins", "FrolovPoints", f), log)
    if not os.path.isfile(os.path.join(dest, "libsasfit" + ext)):
        log(f"warning: no libsasfit{ext} found in {source}")
    log(f"{n} file(s) copied")
    return n
