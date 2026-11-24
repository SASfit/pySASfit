/* shim: replaces the SASfit private.h of plugin thin_objects (tests/studio/port_reference/import_plugin.py) */
#ifndef SHIM_THIN_OBJECTS_PRIVATE_H
#define SHIM_THIN_OBJECTS_PRIVATE_H
#include <sasfit_shim.h>

scalar sasfit_ff_cylsh_sd_bilayergauss(scalar q, sasfit_param * param);
scalar sasfit_ff_cylsh_sd_bilayergauss_f(scalar q, sasfit_param * param);
scalar sasfit_ff_cylsh_sd_bilayergauss_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_cylsh_sd_homoxs(scalar q, sasfit_param * param);
scalar sasfit_ff_cylsh_sd_homoxs_f(scalar q, sasfit_param * param);
scalar sasfit_ff_cylsh_sd_homoxs_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_disc_sd_bilayergauss(scalar q, sasfit_param * param);
scalar sasfit_ff_disc_sd_bilayergauss_f(scalar q, sasfit_param * param);
scalar sasfit_ff_disc_sd_bilayergauss_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_disc_sd_homoxs(scalar q, sasfit_param * param);
scalar sasfit_ff_disc_sd_homoxs_f(scalar q, sasfit_param * param);
scalar sasfit_ff_disc_sd_homoxs_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_ellsh_sd_bilayergauss(scalar q, sasfit_param * param);
scalar sasfit_ff_ellsh_sd_bilayergauss_f(scalar q, sasfit_param * param);
scalar sasfit_ff_ellsh_sd_bilayergauss_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_ellsh_sd_homoxs(scalar q, sasfit_param * param);
scalar sasfit_ff_ellsh_sd_homoxs_f(scalar q, sasfit_param * param);
scalar sasfit_ff_ellsh_sd_homoxs_v(scalar q, sasfit_param * param, int dist);
scalar bilayergauss(scalar nu,sasfit_param * param);
scalar LogNormBiLayerGauss(scalar nu,sasfit_param * param);
scalar sasfit_ff_pcs_bilayergauss(scalar q, sasfit_param * param);
scalar sasfit_ff_pcs_bilayergauss_f(scalar q, sasfit_param * param);
scalar sasfit_ff_pcs_bilayergauss_v(scalar q, sasfit_param * param, int dist);
scalar ell_r(scalar radius, scalar epsilo, scalar t, scalar theta);
scalar pcs_ellcylsh(sasfit_param *param);
double pcs_ellcylsh_cubature(const double *x, size_t ndim, void *pam);
scalar sasfit_ff_pcs_ellcylsh(scalar q, sasfit_param * param);
scalar sasfit_ff_pcs_ellcylsh_f(scalar q, sasfit_param * param);
scalar sasfit_ff_pcs_ellcylsh_v(scalar q, sasfit_param * param, int dist);
scalar homo_cyl_Ampl_core(scalar x, sasfit_param *param);
scalar homo_cyl_I_core(scalar x, sasfit_param *param);
scalar Ampl_homoXS_core(scalar x, sasfit_param *param);
scalar I_homoXS_core(scalar x, sasfit_param *param);
scalar sasfit_ff_pcs_homogeneouscyl(scalar q, sasfit_param * param);
scalar sasfit_ff_pcs_homogeneouscyl_f(scalar q, sasfit_param * param);
scalar sasfit_ff_pcs_homogeneouscyl_v(scalar q, sasfit_param * param, int dist);
scalar homoXScore(scalar x, sasfit_param * param);
scalar homogeneousXS(scalar q, sasfit_param * param);
scalar sasfit_ff_pcs_homogeneousplate(scalar q, sasfit_param * param);
scalar sasfit_ff_pcs_homogeneousplate_f(scalar q, sasfit_param * param);
scalar sasfit_ff_pcs_homogeneousplate_v(scalar q, sasfit_param * param, int dist);
scalar layeredcentrosymmetric_FF(sasfit_param *param);
scalar layeredcentrosymmetric_core(scalar x, sasfit_param *param);
scalar layeredcentrosymmetric_SH(scalar x, sasfit_param * param);
scalar layeredcentrosymmetric_L(scalar q, sasfit_param * param);
int layeredcentrosymmetricxs_cubature(unsigned ndim, const double *x, void *pam, unsigned fdim, double *fval);
scalar sasfit_ff_pcs_layeredcentrosymmetricxs(scalar q, sasfit_param * param);
scalar sasfit_ff_pcs_layeredcentrosymmetricxs_f(scalar q, sasfit_param * param);
scalar sasfit_ff_pcs_layeredcentrosymmetricxs_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_pcs_plate_chains_rw_(scalar q, sasfit_param * param);
scalar sasfit_ff_pcs_plate_chains_rw__f(scalar q, sasfit_param * param);
scalar sasfit_ff_pcs_plate_chains_rw__v(scalar q, sasfit_param * param, int dist);
scalar thincentrosymmetriclayers_core(scalar x, sasfit_param *param);
scalar sasfit_ff_pcs_twoinfinitelythinlayers(scalar q, sasfit_param * param);
scalar sasfit_ff_pcs_twoinfinitelythinlayers_f(scalar q, sasfit_param * param);
scalar sasfit_ff_pcs_twoinfinitelythinlayers_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_rod_sd_ellcylsh(scalar q, sasfit_param * param);
scalar sasfit_ff_rod_sd_ellcylsh_f(scalar q, sasfit_param * param);
scalar sasfit_ff_rod_sd_ellcylsh_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_worm_kholodenko_sd_ellcylsh(scalar q, sasfit_param * param);
scalar sasfit_ff_worm_kholodenko_sd_ellcylsh_f(scalar q, sasfit_param * param);
scalar sasfit_ff_worm_kholodenko_sd_ellcylsh_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_worm_ps1_sd_ellcylsh(scalar q, sasfit_param * param);
scalar sasfit_ff_worm_ps1_sd_ellcylsh_f(scalar q, sasfit_param * param);
scalar sasfit_ff_worm_ps1_sd_ellcylsh_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_worm_ps2_sd_ellcylsh(scalar q, sasfit_param * param);
scalar sasfit_ff_worm_ps2_sd_ellcylsh_f(scalar q, sasfit_param * param);
scalar sasfit_ff_worm_ps2_sd_ellcylsh_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_ff_worm_ps3_sd_ellcylsh(scalar q, sasfit_param * param);
scalar sasfit_ff_worm_ps3_sd_ellcylsh_f(scalar q, sasfit_param * param);
scalar sasfit_ff_worm_ps3_sd_ellcylsh_v(scalar q, sasfit_param * param, int dist);
scalar rod_core(scalar x, sasfit_param *param);
scalar sasfit_sq_p__q___rod(scalar q, sasfit_param * param);
scalar sasfit_sq_p__q___rod_f(scalar q, sasfit_param * param);
scalar sasfit_sq_p__q___rod_v(scalar q, sasfit_param * param, int dist);
scalar disc_core(scalar x, sasfit_param * param);
scalar sasfit_sq_p__q___thin_disc(scalar q, sasfit_param * param);
scalar sasfit_sq_p__q___thin_disc_f(scalar q, sasfit_param * param);
scalar sasfit_sq_p__q___thin_disc_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_sq_p__q___thin_ellipsoidal_shell(scalar q, sasfit_param * param);
scalar sasfit_sq_p__q___thin_ellipsoidal_shell_f(scalar q, sasfit_param * param);
scalar sasfit_sq_p__q___thin_ellipsoidal_shell_v(scalar q, sasfit_param * param, int dist);
scalar XI_CylShell(scalar q, sasfit_param * param);
scalar thinCyl_P_core(scalar alpha, sasfit_param * param);
scalar thinCyl_P(scalar q, sasfit_param * param);
scalar ThinCylShell_H_core(scalar xH, sasfit_param * param);
scalar ThinCylShell_H(scalar q,sasfit_param * param);
scalar ThinCylShell_core(scalar xR, sasfit_param * param);
scalar ThinCylShell(scalar q, sasfit_param * param);
scalar thinCyl_FtoP(sasfit_param * param);
int thin_hollow_cylinder_cubature(unsigned ndim, const double *x, void *pam, unsigned fdim, double *fval);
scalar sasfit_sq_p__q___thin_hollow_cylinder(scalar q, sasfit_param * param);
scalar sasfit_sq_p__q___thin_hollow_cylinder_f(scalar q, sasfit_param * param);
scalar sasfit_sq_p__q___thin_hollow_cylinder_v(scalar q, sasfit_param * param, int dist);
scalar thinF1(scalar q, scalar r, scalar epsilon, scalar alpha);
scalar thinEll_P_core(scalar alpha, sasfit_param * param);
scalar thinEll_FtoP(sasfit_param * param);
int thin_ellip_shell_cubature(unsigned ndim, const double *x, void *pam, unsigned fdim, double *fval);
scalar thin_ellipsoidal_shell(scalar q, scalar R, scalar epsilon, scalar sigma, sasfit_param * param);
scalar thinEll_P(scalar x, sasfit_param * param);
scalar ThinEllShell_R_core(scalar x, sasfit_param * param);
scalar r0(scalar epsilon, scalar s0);
scalar ThinEllShell_S_core(scalar x, sasfit_param * param);
scalar sasfit_sd_LogNorm(scalar, sasfit_param *);
scalar sasfit_sq_p__q___kholodenko_worm(scalar, sasfit_param *);
scalar sasfit_sq_p__q___worm_ps1_(scalar, sasfit_param *);
scalar sasfit_sq_p__q___worm_ps2_(scalar, sasfit_param *);
scalar sasfit_sq_p__q___worm_ps3_(scalar, sasfit_param *);

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
