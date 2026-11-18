"""Compile the SASfit plugin sources in sasfit_src/ unchanged (with the shim headers) into one shared library.

Reference for the Python ports (tests/studio/test_ports.py). Needs a C compiler and GSL (libgsl-dev).
"""

import glob
import os
import shutil
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))


def build(out_dir):
    cc = shutil.which("gcc") or shutil.which("cc")
    if cc is None:
        return None
    lib = os.path.join(out_dir, "libportref.so")
    srcs = [os.path.join(HERE, "shim", "shim.c")] + sorted(
        glob.glob(os.path.join(HERE, "sasfit_src", "*", "*.c"))
    )
    cmd = [
        cc,
        "-O2",
        "-shared",
        "-fPIC",
        "-I",
        os.path.join(HERE, "shim"),
        "-o",
        lib,
        *srcs,
        "-lgsl",
        "-lgslcblas",
        "-lm",
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except (subprocess.CalledProcessError, OSError):
        return None
    return lib
