import numpy as np
from scipy.special import gammaln
from pysasfit.scatter.blocks import BLOCK, z_from_sigma

def brute(d, y, z, s, kind, line=False):
    # gamma pdf of r=R/X (mean 1) weighted by r^s
    r = np.linspace(1e-4, 8, 400001)
    lp = (z+1)*np.log((z+1)) - gammaln(z+1) + z*np.log(r) - (z+1)*r + s*np.log(r)
    w = np.exp(lp - lp.max()); w /= np.trapezoid(w, r)
    out = []
    for yy in y:
        f = BLOCK[d].mono(yy*r, kind)
        out.append(np.trapezoid(f*w, r))
    return np.array(out)

def test_all():
    y = np.array([0.05, 0.5, 2, 6, 15, 30, 80])
    for d in (1, 2, 3):
        for sig in (0.1, 0.3):
            z = z_from_sigma(sig)
            for kind, s in (('P', 2*d), ('F', d)):
                a = BLOCK[d].value(y, z, s, kind)
                b = brute(d, y, z, s, kind)
                err = np.abs(a-b)/np.maximum(np.abs(b), 1e-8)
                print(d, sig, kind, err.max())
                assert np.all(np.abs(a-b) < 2e-5*np.maximum(np.abs(b), 1e-6)+1e-9), (d, sig, kind, err)
