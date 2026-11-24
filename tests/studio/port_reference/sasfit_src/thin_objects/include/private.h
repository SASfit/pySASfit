/* shim: replaces the SASfit private.h of plugin thin_objects (cross-section form factors and the elliptical core-shell cylinders;
 * curated subset, see tests/studio/port_reference/import_plugin.py) */
#ifndef SHIM_THIN_OBJECTS_PRIVATE_H
#define SHIM_THIN_OBJECTS_PRIVATE_H
#include <sasfit_shim.h>

scalar sasfit_ff_pcs_ellcylsh(scalar q, sasfit_param * param);
scalar sasfit_ff_rod_sd_ellcylsh(scalar q, sasfit_param * param);
scalar sasfit_ff_worm_ps1_sd_ellcylsh(scalar q, sasfit_param * param);
scalar sasfit_ff_worm_ps2_sd_ellcylsh(scalar q, sasfit_param * param);
scalar sasfit_ff_worm_ps3_sd_ellcylsh(scalar q, sasfit_param * param);
scalar sasfit_ff_worm_kholodenko_sd_ellcylsh(scalar q, sasfit_param * param);
scalar sasfit_sq_p__q___rod(scalar q, sasfit_param * param);
scalar sasfit_ff_pcs_homogeneouscyl(scalar q, sasfit_param * param);
scalar sasfit_ff_pcs_homogeneousplate(scalar q, sasfit_param * param);
scalar sasfit_ff_pcs_twoinfinitelythinlayers(scalar q, sasfit_param * param);
scalar sasfit_ff_pcs_bilayergauss(scalar q, sasfit_param * param);
scalar sasfit_ff_pcs_bilayergauss_f(scalar q, sasfit_param * param);
scalar sasfit_ff_pcs_layeredcentrosymmetricxs(scalar q, sasfit_param * param);
scalar sasfit_ff_pcs_plate_chains_rw_(scalar q, sasfit_param * param);
scalar sasfit_ff_disc_sd_homoxs(scalar q, sasfit_param * param);
scalar sasfit_ff_disc_sd_bilayergauss(scalar q, sasfit_param * param);
scalar sasfit_sq_p__q___thin_disc(scalar q, sasfit_param * param);
scalar sasfit_sq_p__q___kholodenko_worm(scalar q, sasfit_param * param);
scalar sasfit_sq_p__q___worm_ps1_(scalar q, sasfit_param * param);
scalar sasfit_sq_p__q___worm_ps2_(scalar q, sasfit_param * param);
scalar sasfit_sq_p__q___worm_ps3_(scalar q, sasfit_param * param);
scalar sasfit_sd_LogNorm(scalar x, sasfit_param *param);
scalar find_LogNorm_int_range(scalar dim, scalar x0, scalar sigma, scalar *Xstart, scalar *Xend, sasfit_param *param);

//
// add local defines here:
// #define P0 param->p[0]
//

typedef scalar sasfit_cubature_f(sasfit_param *);

typedef struct
{
	sasfit_param *param;
	sasfit_cubature_f *func; //!< Selects the kernel function to use, when used with gsl functions.
	scalar *cubxmin;
	scalar *cubxmax;
	int ndim;
} cubature_param;


#define R_SQ_ELLSH	param->p[0]
#define EPSILON_SQ_ELLSH	param->p[1]
#define SIGMA_R_SQ_ELLSH	param->p[2]

#define LB		param->p[0]
#define B		param->p[0]
#define L		param->p[1]
#define EXVOL	param->p[2]
#define BETA	param->p[3]
#define RC  	param->p[4]
#define LP  	param->p[4]
#define RP  	param->p[4]
#define MONO_SD param->p[5]

#define T_ELLSH	        param->p[0]
#define SIGMA_T_ELLSH	fabs(param->p[1])
#define R0_ELLSH	    param->p[2]
#define SIGMA_R0_ELLSH	fabs(param->p[3])
#define EPSILON_ELLSH	param->p[4]



#define Q	param->p[MAXPAR-1]
#define P	param->p[MAXPAR-2]

#define NU1	        param->p[MAXPAR-3]
#define LNDISTR1	param->p[MAXPAR-4]
#define NU1MIN      param->p[MAXPAR-5]
#define NU1MAX	    param->p[MAXPAR-6]

#define NU2	        param->p[MAXPAR-7]
#define LNDISTR2	param->p[MAXPAR-8]
#define NU2MIN	    param->p[MAXPAR-9]
#define NU2MAX	    param->p[MAXPAR-10]

#define ALPHA   param->p[MAXPAR-11]
#define CLB		param->p[MAXPAR-12]
#define	N	    param->p[MAXPAR-13]

scalar fRg2(scalar, scalar);
scalar falpha2(scalar, scalar);
scalar IDebye(scalar, scalar);
scalar SB_term(scalar);
scalar S_SB(scalar, scalar, sasfit_param *);
scalar S_EXV(scalar, scalar, sasfit_param *);
scalar fcorr (scalar, scalar, sasfit_param *);
scalar IRod(scalar, scalar);
scalar ThinEllShell_R_core(scalar, sasfit_param *);
scalar ThinEllShell_S_core(scalar, sasfit_param *);
scalar thin_ellipsoidal_shell(scalar, scalar, scalar, scalar, sasfit_param *);
scalar PS3_EXP_SD(scalar, sasfit_param *);
#endif
