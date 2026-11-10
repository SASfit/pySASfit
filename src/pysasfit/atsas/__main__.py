"""python -m pysasfit.atsas [--selftest] [--atsas-dir DIR] [data files]"""
import sys

from .core import _main

sys.exit(_main())
