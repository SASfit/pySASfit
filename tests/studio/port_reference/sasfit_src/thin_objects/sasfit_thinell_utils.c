/* thinF1 ... ThinEllShell_S_core of thin_objects/sasfit_ao_utils.c (unchanged). Reference copy for the Python ports. */
#include "include/private.h"

scalar thinF1(scalar q, scalar r, scalar epsilon, scalar alpha)
{
    scalar u, Rtmp;

	Rtmp = r*gsl_hypot(sin(alpha),epsilon*cos(alpha));

	u = q * Rtmp;

    return gsl_sf_bessel_j0(u);
}


scalar thinEll_P_core(scalar alpha, sasfit_param * param)
{
	scalar S, ae, lEPSILON, lR,lQ;

	lEPSILON = param->p[1];
	lQ		 = param->p[MAXPAR-1];
	lR		 = param->p[MAXPAR-2];

	SASFIT_ASSERT_PTR(param);

	if (lEPSILON < 1.)
	{
		ae = acos(lEPSILON);
		S = 2.*M_PI*lR*lR*(1.+lEPSILON*gsl_atanh(sin(ae))/sin(ae));
	}
	else if (lEPSILON > 1.)
	{
		ae = acos(1./lEPSILON);
		S = 2.*M_PI*lR*lR*(1.+lEPSILON*ae/tan(ae));
	} else
	{
		S = 4.*M_PI*lR*lR;
	}
	return  pow(S * thinF1(lQ, lR, lEPSILON, alpha),2.0)*sin(alpha);
}

scalar thinEll_FtoP(sasfit_param * param)
{
	scalar S, ae, lEPSILON, lR,lQ;

	lEPSILON = EPSILON_SQ_ELLSH;
	lQ		 = Q;
	lR		 = R_SQ_ELLSH*NU1;

	SASFIT_ASSERT_PTR(param);

	if (lEPSILON < 1.)
	{
		ae = acos(lEPSILON);
		S = 2.*M_PI*lR*lR*(1.+lEPSILON*gsl_atanh(sin(ae))/sin(ae));
	}
	else if (lEPSILON > 1.)
	{
		ae = acos(1./lEPSILON);
		S = 2.*M_PI*lR*lR*(1.+lEPSILON*ae/tan(ae));
	} else
	{
		S = 4.*M_PI*lR*lR;
	}
	return  LNDISTR1*gsl_pow_int(S * thinF1(lQ, lR, lEPSILON, ALPHA),lround(P))*sin(ALPHA);
}

int thin_ellip_shell_cubature(unsigned ndim, const double *x, void *pam,
      unsigned fdim, double *fval) {
	sasfit_param subParam;
	sasfit_init_param( &subParam );
	sasfit_param *param;
	cubature_param *cparam;
	cparam = (cubature_param *) pam;
	param = cparam->param;

	fval[0] = 0;
	if ((ndim < 1) || (fdim < 1)) {
		sasfit_out("false dimensions fdim:%d ndim:%d\n",fdim,ndim);
		return 1;
	}
	if ((ndim == 1) && (SIGMA_R_SQ_ELLSH==0)) {
		LNDISTR1 = 1;
		NU1 = 1;
		ALPHA=x[0];
	} else if ((ndim == 1) && (EPSILON_SQ_ELLSH==1)) {
		subParam.p[0] = 1.0;
		subParam.p[1] = SIGMA_R_SQ_ELLSH;
		subParam.p[2] = 1.0;
		subParam.p[3] = 1.0;
		NU1=x[0];
	    ALPHA = M_PI_2;
		LNDISTR1 = sasfit_sd_LogNorm(NU1, &subParam);
		SASFIT_CHECK_SUB_ERR(param, subParam);
		if ( subParam.errStatus != FALSE ) {
			sasfit_out("LogNormError: SIGMA:%lf\n",SIGMA_R_SQ_ELLSH);
			return 1;
		}
	} else {
		subParam.p[0] = 1.0;
		subParam.p[1] = SIGMA_R_SQ_ELLSH;
		subParam.p[2] = 1.0;
		subParam.p[3] = 1.0;
		ALPHA = x[0];
		NU1 = x[1];
		LNDISTR1 = sasfit_sd_LogNorm(NU1, &subParam);
		SASFIT_CHECK_SUB_ERR(param, subParam);
		if ( subParam.errStatus != FALSE ) {
			sasfit_out("LogNormError: SIGMA:%lf\n",SIGMA_R_SQ_ELLSH);
			return 1;
		}
	}
    fval[0] = (cparam->func)(param);
    return 0;
}

scalar thin_ellipsoidal_shell(scalar q, scalar R, scalar epsilon, scalar sigma, sasfit_param * param)
{
	scalar res,err,sum, Rmin, Rmax;
    scalar cubxmin[2], cubxmax[2], fval[1], ferr[1];
    int intstrategy, ndim;
	cubature_param cparam;

	SASFIT_ASSERT_PTR(param); // assert pointer param is valid
    EPSILON_SQ_ELLSH = epsilon;
    R_SQ_ELLSH = R;
    SIGMA_R_SQ_ELLSH = sigma;
	NU1 = 1;
	LNDISTR1=1.0;
	ndim  =2;
	if (SIGMA_R_SQ_ELLSH==0) {
		ndim=1;
		NU1MIN = 1;
		NU1MAX = 1;
	} else {
		ndim  =2;
		find_LogNorm_int_range(2*P,1.0,SIGMA_R_SQ_ELLSH,&Rmin, &Rmax, param);
		NU1MIN=Rmin;
		NU1MAX=Rmax;
	}
	if (EPSILON_SQ_ELLSH==1) {
        ndim=ndim-1;
        ALPHA = M_PI_2;
        cubxmin[0]=NU1MIN;
        cubxmax[0]=NU1MAX;
	} else {
	    cubxmin[0]=0;
        cubxmax[0]=M_PI_2;
        cubxmin[1]=NU1MIN;
        cubxmax[1]=NU1MAX;
	}
    if (ndim==0) {
        NU1 = 1;
        LNDISTR1=1.0;
        ALPHA = M_PI_2;
        return thinEll_FtoP(param);
    }
	cparam.ndim=ndim;
	cparam.func = &thinEll_FtoP;
    cparam.param = param;

	intstrategy = sasfit_get_int_strategy();
	intstrategy=P_CUBATURE;
	switch(intstrategy) {
    case H_CUBATURE: {
			hcubature(1, &thin_ellip_shell_cubature,&cparam,ndim, cubxmin, cubxmax,
				100000, 0.0, sasfit_eps_get_nriq(), ERROR_PAIRED,
				fval, ferr);
			sum = fval[0];
            break;
            }
    case P_CUBATURE: {
			pcubature(1, &thin_ellip_shell_cubature,&cparam,ndim, cubxmin, cubxmax,
				100000, 0.0, sasfit_eps_get_nriq(), ERROR_PAIRED,
				fval, ferr);
			sum = fval[0];
            break;
            }
    default: {
//		    sasfit_out("ise default sasfit_integrate routine\n");
            pcubature(1, &thin_ellip_shell_cubature,&cparam,ndim, cubxmin, cubxmax,
				100000, 0.0, sasfit_eps_get_nriq(), ERROR_PAIRED,
				fval, ferr);
			sum = fval[0];
            break;
            }
    }
	return sum;
}

scalar thinEll_P(scalar x, sasfit_param * param)
{
	scalar lEPSILON;
	lEPSILON = param->p[1];

	if (x==0) return 0.0;
	param->p[MAXPAR-2]=x;

	SASFIT_ASSERT_PTR(param);
	if (lEPSILON == 1) {
		return thinEll_P_core(M_PI/2.0,param);
	} else {
		return sasfit_integrate(0.0, M_PI/2.0, &thinEll_P_core, param);
	}
}

scalar ThinEllShell_R_core(scalar x, sasfit_param * param)
{
	scalar IP, LNdistr, lR,lQ, lSIGMA_R, lEPSILON;
	sasfit_param subParam;

	if (x==0) return 0.0;

	lR = param->p[0];
	lEPSILON = param->p[1];
	lSIGMA_R = param->p[2];
	lQ = param->p[MAXPAR-1];

	SASFIT_ASSERT_PTR(param);

	IP = thinEll_P(x, param);

	if (lSIGMA_R != 0) {
		sasfit_init_param( &subParam );
		subParam.p[0] = 1.0;
		subParam.p[1] = lSIGMA_R;
		subParam.p[2] = 1.0;
		subParam.p[3] = lR;
		LNdistr = sasfit_sd_LogNorm(x, &subParam);
		SASFIT_CHECK_SUB_ERR(param, subParam);
	} else {
		LNdistr = 1;
	}

	return LNdistr*IP;
}


scalar r0(scalar epsilon, scalar s0)
{
	scalar r0, ae;
	if (epsilon < 1.)
	{
		ae = acos(epsilon);
		r0 = sqrt(s0/(1.+epsilon*gsl_atanh(sin(ae))/sin(ae)) / (2.*M_PI));
	}
	else if (epsilon > 1.)
	{
		ae = acos(1./epsilon);
		r0 = sqrt(s0/(1.+epsilon*ae/tan(ae)) / (2.*M_PI));
	} else
	{
		r0 = sqrt(s0 / 4./M_PI);
	}
	return r0;
}

scalar ThinEllShell_S_core(scalar x, sasfit_param * param)
{
	scalar IP, LNdistr, lS, lQ, lSIGMA_S, lEPSILON;
	sasfit_param subParam;

	lS = param->p[0];
	lEPSILON = param->p[1];
	lSIGMA_S = param->p[2];
	lQ = param->p[MAXPAR-1];

	SASFIT_ASSERT_PTR(param);

	param->p[MAXPAR-2] = r0(lEPSILON, lS);

	IP = thinEll_P(lQ, param);

	sasfit_init_param( &subParam );
	subParam.p[0] = 1.0;
	subParam.p[1] = lSIGMA_S;
	subParam.p[2] = 1.0;
	subParam.p[3] = lS;

	LNdistr = sasfit_sd_LogNorm(x, &subParam);
	SASFIT_CHECK_SUB_ERR(param, subParam);

	return LNdistr*IP;
}

