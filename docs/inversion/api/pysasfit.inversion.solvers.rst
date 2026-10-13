pysasfit.inversion.solvers
==========================

The individual solver implementations. Most callers should go through
:mod:`pysasfit.inversion.solver_registry` (:ref:`solver-registry`) instead of
these modules directly.

pysasfit.inversion.solvers.base
-------------------------------

.. automodule:: pysasfit.inversion.solvers.base
   :members:

pysasfit.inversion.solvers.acceleration
---------------------------------------

Biggs-Andrews acceleration, applied by default (``accelerate=True``)
across the EM-family solvers below.

.. automodule:: pysasfit.inversion.solvers.acceleration
   :members:

pysasfit.inversion.solvers.em
-----------------------------

.. automodule:: pysasfit.inversion.solvers.em
   :members:

pysasfit.inversion.solvers.em_general
-------------------------------------

EM for signed kernels (Chae, Martin & Walker, 2019, Sec. 6) with
:math:`\chi^2` row weighting, acceleration safeguard and smoothing or
positive/negative-entropy MaxEnt regularization -- required for the
``4*pi*j0(qr)`` PDDF kernel, since the ordinary EM update above is only
valid for a non-negative kernel. See :doc:`../theory_em_maxent`.

.. automodule:: pysasfit.inversion.solvers.em_general
   :members:

pysasfit.inversion.solvers.maxent
---------------------------------

.. automodule:: pysasfit.inversion.solvers.maxent
   :members:

pysasfit.inversion.solvers.hansen_maxent
----------------------------------------

.. automodule:: pysasfit.inversion.solvers.hansen_maxent
   :members:

pysasfit.inversion.solvers.svd_methods
--------------------------------------

Tikhonov/TSVD via SVD, plus GCV-based parameter selection.

.. automodule:: pysasfit.inversion.solvers.svd_methods
   :members:

pysasfit.inversion.solvers.general_tikhonov
-------------------------------------------

.. automodule:: pysasfit.inversion.solvers.general_tikhonov
   :members:

pysasfit.inversion.solvers.bayesian_evidence
--------------------------------------------

Bayesian evidence maximization (Vestergaard & Hansen, 2006) for choosing
the regularization weight -- generic over the regularization operator
``L``; also the Laplace posterior covariance. The two-hyperparameter block
extension (``log_evidence_blocks`` / ``evidence_search_blocks``) belongs
to the retired two-segment solver (see :doc:`../theory_investigation`).

.. automodule:: pysasfit.inversion.solvers.bayesian_evidence
   :members:

pysasfit.inversion.solvers.hansen_ift -- moved to ``_obsolete/``
----------------------------------------------------------------

Not imported by ``solver_registry.py`` or any test -- an orphaned leftover
from development, moved to
``_obsolete/src/pysasfit.inversion/solvers/hansen_ift.py`` (see
``_obsolete/README.md``) rather than deleted outright. Hansen's
boundary-constrained operator itself is not gone -- it lives on in
:func:`pysasfit.inversion.regularization.hansen_smoothness_cholesky`, used by
``bayesian_evidence`` with ``penalty="hansen_printed"``.

pysasfit.inversion.solvers.lambda_search
----------------------------------------

Discrepancy-principle and L-curve searches, used by the EM-family
solvers in ``solver_registry.py``.

.. automodule:: pysasfit.inversion.solvers.lambda_search
   :members:

pysasfit.inversion.solvers.arls
-------------------------------

.. automodule:: pysasfit.inversion.solvers.arls
   :members:
