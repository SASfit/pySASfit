Size distribution / p(r): ``pysasfit.inversion``
================================================

Library (+ a Qt GUI) for solving Fredholm integrals of
the first kind arising in small-angle scattering: recovering a
**pair-distance distribution function (PDDF)** p(r) or a **size
distribution** from measured I(q).

Features: EM, EM + smoothing and EM + MaxEnt (SASfit), EM for signed
kernels with smoothing or MaxEnt, Bayesian-evidence IFT with Hansen's
penalties, Tikhonov/TSVD with GCV, ARLS, Hansen's MaxEnt; point grids
(uniform, quadratic, logarithmic) with linear or cubic interpolation, or a
cubic B-spline basis for any solver; analytic and bootstrap uncertainties.
Pair-distance kernels for globular particles, the cross-section of long
cylinders and the thickness of lamellae (Hansen 2016); data without error
bars (error models, noise-scale estimate); project files and the last
session (File and Settings menus).

.. toctree::
   :maxdepth: 2
   :caption: Contents

   installation
   usage
   gui
   theory
   references

The API reference of this tool is in the :doc:`API section <../api/index>`.

