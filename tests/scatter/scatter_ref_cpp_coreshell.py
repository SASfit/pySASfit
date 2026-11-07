import numpy as np
from pysasfit.scatter.coreshell import CoreShellSphere
from pysasfit.scatter.blocks import z_from_sigma
from scatter_testutil import gam_w
from pysasfit.scatter.blocks import BLOCK

def cpp_cs(q, sigma, Rout, p1, rho, fix_ac9=False):
    zr = (1-sigma**2)/sigma**2; zz = zr
    radius = p1*Rout; radiusm = Rout
    cc1=rho**2; cc2=2*p1*rho*(1-rho); cc3=(1-rho)**2*p1**2; cc4=-2*rho**2
    cc5=-2*p1*rho*(1-rho); cc6=rho**2; cc7=-2*rho*(1-rho); cc8=-(1-rho)**2*2*p1
    cc9=2*rho*(1-rho); cc10=(1-rho)**2
    vv3=((1-rho)*p1**3+rho)**2
    argq=q*radiusm/(zz+1); argpq=q*radius/(zz+1)
    pqr=(1/(2.0*zr*(zr-1)*(zr-2)*(zr-3)))*argq**-4
    at=np.arctan
    ac3=pqr*(1+np.cos((zr-3)*at(2*argpq))/(1+4*argpq**2)**((zr-3)/2))
    ac8=(pqr/((zr-4)*argq))*np.sin((zr-4)*at(2*argpq))/(1+4*argpq**2)**((zr-4)/2)
    ac10=(pqr/((zr-4)*(zr-5)*argq**2))*(1-np.cos((zr-5)*at(2*argpq))/(1+4*argpq**2)**((zr-5)/2))
    F121=9*(cc3*ac3+cc8*ac8+cc10*ac10)/vv3
    argbm=(zr-3)*at(argpq-argq); nenbm=(1+(argpq-argq)**2)**((zr-3)/2)
    argbp=(zr-3)*at(argpq+argq); nenbp=(1+(argpq+argq)**2)**((zr-3)/2)
    ac2=pqr*(np.cos(argbm)/nenbm+np.cos(argbp)/nenbp)
    argep=(zr-4)*at(argpq+argq); nenep=(1+(argpq+argq)**2)**((zr-4)/2)
    argem=(zr-4)*at(argpq-argq); nenem=(1+(argpq-argq)**2)**((zr-4)/2)
    ac5=(pqr/((zr-4)*argq))*(np.sin(argep)/nenep-np.sin(argem)/nenem)
    arggp=(zr-4)*at(argpq+argq); nengp=(1+(argpq+argq)**2)**((zr-4)/2)
    arggm=(zr-4)*at(argq-argpq); nengm=(1+(argq-argpq)**2)**((zr-4)/2)
    ac7=(pqr/((zr-4)*argq))*(np.sin(arggp)/nengp-np.sin(arggm)/nengm)
    if fix_ac9:   # C++ reuses the x^-4 phase/damping (zr-3) here; x^-6 needs (zr-5)
        ac9=(pqr/((zr-4)*(zr-5)*argq**2))*(np.cos((zr-5)*at(argpq-argq))/(1+(argpq-argq)**2)**((zr-5)/2)-np.cos((zr-5)*at(argpq+argq))/(1+(argpq+argq)**2)**((zr-5)/2))
    else:
        ac9=(pqr/((zr-4)*(zr-5)*argq**2))*(np.cos(argbm)/nenbm-np.cos(argbp)/nenbp)
    F122=9*(cc2*ac2+cc5*ac5+cc7*ac7+cc9*ac9)/vv3
    ac1=pqr*(1+np.cos((zr-3)*at(2*argq))/(1+4*argq**2)**((zr-3)/2))
    ac4=(pqr/((zr-4)*argq))*np.sin((zr-4)*at(2*argq))/(1+4*argq**2)**((zr-4)/2)
    ac6=(pqr/((zr-4)*(zr-5)*argq**2))*(1-np.cos((zr-5)*at(2*argq))/(1+4*argq**2)**((zr-5)/2))
    F123=9*(cc1*ac1+cc4*ac4+cc6*ac6)/vv3
    return F121+F122+F123

if __name__ == "__main__":
    sg, Rout, p1, rho = 0.1, 40., 0.6, 0.5
    q = np.array([0.05, 0.1, 0.2, 0.4, 0.8])
    ref = cpp_cs(q, sg, Rout, p1, rho)
    class S0(CoreShellSphere): pass
    mine = BLOCK  # placeholder
    from pysasfit.scatter.blocks import cross3
    z = z_from_sigma(sg)
    c = np.array([(1-rho)*p1**3, rho])
    pp = [p1, 1.0]
    tot = 0
    for i in range(2):
        for j in range(i+1):
            tot = tot + c[i]*c[j]*(1 if i==j else 2)*cross3(pp[i], pp[j], q*Rout, z, 0)
    mine0 = tot/c.sum()**2
    r, w = gam_w(sg, 0)
    br = np.array([np.trapezoid(((c[0]*BLOCK[3].mono(qq*Rout*r*p1,'F')+c[1]*BLOCK[3].mono(qq*Rout*r,'F'))/c.sum())**2*w, r) for qq in q])
    np.set_printoptions(precision=5, linewidth=160)
    print("C++ ", ref); print("mine", mine0); print("brut", br)
    print("rel C++/brute", ref/br-1); print("rel mine/brute", mine0/br-1)
