Installation
============

``pysasfit.inversion`` is part of pySASfit. From the repository root:

.. code-block:: bash

   pip install -e ".[gui,dev]"   # library + Qt GUI + pytest
   pip install -e .              # library only (numpy, scipy)

Python >= 3.10, numpy, scipy; PySide6 and matplotlib for the GUI. Start the
tool with ``pysasfit-inversion``, or ``pysasfit-gui`` for all tools as tabs.
