pysasfit.inversion
==================

The library's top-level modules. Most of the day-to-day interface is
:data:`pysasfit.inversion.solver_registry.SOLVER_REGISTRY` and
:data:`pysasfit.inversion.kernel_registry.KERNEL_REGISTRY` -- see
:doc:`../usage`.

pysasfit.inversion
------------------

.. automodule:: pysasfit.inversion
   :members:

pysasfit.inversion.io_utils
---------------------------

.. automodule:: pysasfit.inversion.io_utils
   :members:

pysasfit.inversion.background
-----------------------------

.. automodule:: pysasfit.inversion.background
   :members:

pysasfit.inversion.kernels
--------------------------

.. automodule:: pysasfit.inversion.kernels
   :members:

pysasfit.inversion.kernel_registry
----------------------------------

.. automodule:: pysasfit.inversion.kernel_registry
   :members:

pysasfit.inversion.regularization
---------------------------------

Regularization operators: identity, first and second difference,
Hansen's (2000) boundary-constrained matrix (``hansen_smoothness_cholesky``,
the "printed" variant; eq. 19 itself is built in
``solver_registry._hansen_eq19_factor``), and the retired Hann taper and
two-segment split -- see :doc:`../theory_regularization`.

.. automodule:: pysasfit.inversion.regularization
   :members:

pysasfit.inversion.spline_basis
-------------------------------

Clamped cubic B-spline basis with :math:`B(0)=B(r_{max})=0` and uniform,
quadratic or logarithmic knots -- see :doc:`../theory_discretization`.

.. automodule:: pysasfit.inversion.spline_basis
   :members:

pysasfit.inversion.goodness_of_fit
----------------------------------

Error models for data without uncertainties and alternative fit
statistics -- see :doc:`../theory_forward_model`.

.. automodule:: pysasfit.inversion.goodness_of_fit
   :members:

pysasfit.inversion.uncertainty
------------------------------

Parametric bootstrap -- see :doc:`../theory_uncertainty`.

.. automodule:: pysasfit.inversion.uncertainty
   :members:

pysasfit.inversion.lambda_selection
-----------------------------------

Standalone L-curve (Menger-curvature) regularization-parameter selection.
Not imported by ``solver_registry.py`` directly, but it IS imported by
:mod:`pysasfit.inversion.solvers.lambda_search` (``find_corner``,
``LCurvePoint``), which ``solver_registry.py`` uses throughout -- so this
module is a required dependency, not an unused leftover (an earlier pass
of this documentation incorrectly flagged it as obsolete and moved it
out, which broke the import chain; it has been restored).

.. automodule:: pysasfit.inversion.lambda_selection
   :members:

.. _solver-registry:

pysasfit.inversion.solver_registry
----------------------------------

The GUI-facing, auto-tuned solvers. ``run_solver(key, A, b, dI,
spline=None, **options)`` runs any of them, optionally on a B-spline basis
(``SplineOptions``). Every entry of ``SOLVER_REGISTRY`` appears in the GUI;
``SolverSpec.options`` become drop-downs. ``SOLVER_ALIASES`` and
``RETIRED_SOLVERS`` keep old keys working. See :doc:`../usage` for the list
of solvers and options.

.. automodule:: pysasfit.inversion.solver_registry
   :members:
