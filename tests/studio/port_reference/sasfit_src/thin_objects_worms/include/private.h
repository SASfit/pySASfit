/* shim: replaces the SASfit private.h of plugin thin_objects_worms (tests/studio/port_reference/import_plugin.py) */
#ifndef SHIM_THIN_OBJECTS_WORMS_PRIVATE_H
#define SHIM_THIN_OBJECTS_WORMS_PRIVATE_H
#include <sasfit_shim.h>

scalar sasfit_sq_freely_joined_chain_of_rods(scalar q, sasfit_param * param);
scalar sasfit_sq_freely_joined_chain_of_rods_f(scalar q, sasfit_param * param);
scalar sasfit_sq_freely_joined_chain_of_rods_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_sq_Pprime_KholodenkoWorm_core(scalar x, sasfit_param *param);
scalar sasfit_sq_p__q___kholodenko_worm(scalar q, sasfit_param * param);
scalar sasfit_sq_p__q___kholodenko_worm_f(scalar q, sasfit_param * param);
scalar sasfit_sq_p__q___kholodenko_worm_v(scalar q, sasfit_param * param, int dist);
scalar sasfit_sq_p__q___worm_ps1_(scalar q, sasfit_param * param);
scalar sasfit_sq_p__q___worm_ps1__f(scalar q, sasfit_param * param);
scalar sasfit_sq_p__q___worm_ps1__v(scalar q, sasfit_param * param, int dist);
scalar sasfit_sq_p__q___worm_ps2_(scalar q, sasfit_param * param);
scalar sasfit_sq_p__q___worm_ps2__f(scalar q, sasfit_param * param);
scalar sasfit_sq_p__q___worm_ps2__v(scalar q, sasfit_param * param, int dist);
scalar sasfit_sq_p__q___worm_ps3_(scalar q, sasfit_param * param);
scalar sasfit_sq_p__q___worm_ps3__f(scalar q, sasfit_param * param);
scalar sasfit_sq_p__q___worm_ps3__v(scalar q, sasfit_param * param, int dist);
scalar sasfit_sq_koyama_worms(scalar q, sasfit_param * param);
scalar sasfit_sq_koyama_worms_f(scalar q, sasfit_param * param);
scalar sasfit_sq_koyama_worms_v(scalar q, sasfit_param * param, int dist);
/* the Koyama file defines global helpers whose names collide with polymer_stars in the combined library */
#define xg koyama_xg
#define xf koyama_xf
#define phi koyama_phi
#define x_func koyama_x_func
#define root_x koyama_root_x
#define which_x koyama_which_x
scalar IDebye(scalar q, scalar Rg);
scalar S_SB(scalar q, scalar Rg, sasfit_param *param);
scalar S_EXV(scalar q, scalar Rg, sasfit_param *param);
scalar fcorr(scalar q, scalar RG, sasfit_param *param);
scalar IRod(scalar q, scalar H);

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
