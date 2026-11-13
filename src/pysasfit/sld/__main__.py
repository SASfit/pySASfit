"""python -m pysasfit.sld [formula] [options]   (no arguments: the window)"""
import sys

from .cli import main

sys.exit(main())
