"""Fast paths of core.background (flat background, packed Jacobian) equal the general evaluation."""
import numpy as np

from pysasfit.core.background import Background, Constant, OrnsteinZernike, PowerLaw


def _general_value(bg, q, z):
    th = bg.theta(z)
    out, i = np.zeros_like(q), 0
    for t in bg.terms:
        k = len(t.names)
        out = out + t.f(q, th[i:i + k])
        i += k
    return out


def test_fast_paths():
    q = np.linspace(0.01, 0.3, 50)
    I = 1 / q
    for bg in (Background([Constant()]), Background([Constant(), Constant(0.2, fit=False)]),
               Background([Constant(), PowerLaw(), OrnsteinZernike(fit_xi=False)])):
        bg.setup(q, I)
        z = bg.z0() + 0.1
        assert np.array_equal(bg.value(q, z), _general_value(bg, q, z))
        J = bg.jac(q, z)
        h = 1e-6
        for c in range(J.shape[1]):
            dz = np.zeros_like(z)
            dz[c] = h
            num = (bg.value(q, z + dz) - bg.value(q, z - dz)) / (2 * h)
            assert np.allclose(J[:, c], num, rtol=1e-5, atol=1e-8 * np.abs(num).max())
