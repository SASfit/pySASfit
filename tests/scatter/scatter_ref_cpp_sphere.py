import numpy as np
from math import pi, sqrt, gamma, factorial
from pysasfit.scatter.blocks import BLOCK, z_from_sigma
from pysasfit.scatter.shapes import Sphere
from scatter_testutil import gam_w

def carr4p(z, R, nmax=120):
    xr2z = (R/(2*(z+1)))**2
    z12 = 1.0; fk = 1.0; gam3 = sqrt(pi)/2; xr = 1.0; out=[1.0]
    for n in range(1, nmax+1):
        z12 *= ((z+1)-2+2*n)*((z+1)-1+2*n); fk *= n; gam3 *= (2*n+1)/2.0; xr = -xr*xr2z
        out.append(9*sqrt(pi)*4.0**n*z12*xr/(2.0*(n+3)*(n+2)*(n+1.5)*gam3*fk))
    return np.array(out)

def ref_P(q, sigma, R, limfac=0.4, buggy=True):
    zr = (1-sigma**2)/sigma**2
    c = carr4p(zr, R)
    n4 = next((n for n in range(1, 121) if abs(c[n]) < 1e-30), 120)
    while n4 > 1 and (not np.isfinite(c[n4]) or abs(c[n4]) < 1e-99): n4 -= 1
    limq4 = abs(c[n4])**(-1/(2.0*n4))
    out = []
    for qq in q:
        if qq < limfac*limq4:
            s = 1.0; old = 0.0; qq2 = 1.0
            for n in range(1, 101):
                qq2 *= qq*qq; s += c[n]*qq2
                if abs((s-old)/s) < 1e-4: break
                old = s
            out.append(s)
        else:
            a = qq*R/(zr+1)
            pqr = (1/(2.0*zr*(zr-1)*(zr-2)*(zr-3)))*a**-4
            t = np.arctan(2*a); b = 1+4*a*a
            pq1 = pqr*(1+np.cos((zr-3)*t)/b**((zr-3)/2))
            pq2 = pqr/((zr-4)*a)*np.sin((zr-4)*t)/b**((zr-4)/2)
            if buggy: pq3 = pqr/((zr-4)*(zr-5)*a*a)*(1-np.cos((zr-5)*t)/b**(zr-5)/2.0)
            else:     pq3 = pqr/((zr-4)*(zr-5)*a*a)*(1-np.cos((zr-5)*t)/b**((zr-5)/2.0))
            out.append(9*(pq1-2*pq2+pq3))
    return np.array(out), limq4
if __name__ == "__main__":
    R = 50.; sg = 0.1; z = z_from_sigma(sg)
    q = np.array([0.001, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5])
    rf, lim = ref_P(q, sg, R)
    rf2, _ = ref_P(q, sg, R, buggy=False)
    mine0 = BLOCK[3].value(q*R, z, 0, 'P')
    mine6 = Sphere(R, sg).P(q)
    r, w = gam_w(sg, 0)
    brute0 = np.array([np.trapezoid(BLOCK[3].mono(qq*R*r,'P')*w, r) for qq in q])
    print("limq4", lim, "-> switch q =", 0.4*lim)
    np.set_printoptions(precision=5, linewidth=160)
    print("q            ", q)
    print("C++ port     ", rf)
    print("C++ (pq3 fix)", rf2)
    print("mine s=0     ", mine0)
    print("brute s=0    ", brute0)
    print("mine s=6     ", mine6)
    print("rel C++/brute0", rf/brute0-1)
    print("rel fix/brute0", rf2/brute0-1)
    print("rel mine0/brute0", mine0/brute0-1)
