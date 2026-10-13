"""Start the pySASfit launcher without installing: python run_gui.py [datafile]"""

import os
import sys

try:
    import pysasfit  # noqa: F401
except ImportError:
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from pysasfit.gui.launcher import main

if __name__ == "__main__":
    sys.exit(main())
