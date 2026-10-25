# SPDX-License-Identifier: BSD-3-Clause
"""pySASred: SANS data reduction for SINQ SANS-I (SANS-LLB later) with exact
first-order error propagation from every independent source.

data.uarray      ``UArray``: value + Jacobian to each named source (sparse + low rank)
core             counting error, dead time, attenuator, monitor/time, geometry and
                 solid angle (detector tilt), beam centre, radial/sector average, Qdev
io.sinq          SANS-I HDF reader (units from attributes, strict required fields)
io.nxcansas      NXcanSAS 1D output with Idev, covariance, systematic terms, budget
io.ascii         q I Idev Qdev text output
catalogue        header table of a raw-data folder
reduce           ``reduce_run_1d``: one run to I(q) in counts/monitor/sr
gui              Qt tool (``pysasfit-sasred`` and the "SANS reduction" launcher tab)

Needs ``h5py`` and ``pyyaml`` (``pip install pysasfit[sasred]``).
"""
__version__ = "0.1.0.dev0"
