"""Multicomponent fixed-point operator: unique-pair transforms and the compiled batched solve give the
converged solution of the LAPACK path."""
import numpy as np

from test_multicomponent import _yukawa_solver


def test_compiled_solve_same_fixed_point(monkeypatch):
    from pysasfit.oz.picardOZsolver import PicardOZsolver
    a = np.asarray(_yukawa_solver("doHNCclosure", None, maxit=2000).getRDF())
    monkeypatch.setattr(PicardOZsolver, "compiledLinearAlgebra", False, raising=False)
    b = np.asarray(_yukawa_solver("doHNCclosure", None, maxit=2000).getRDF())
    assert np.allclose(a, b, rtol=1e-9, atol=1e-12)
