/*
 * Author(s) of this file:
 *   Joachim Kohlbrecher (joachim.kohlbrecher@psi.ch)
 */

#include "include/private.h"
#include <sasfit_error_ff.h>

// define shortcuts for local parameters/variables

#define RTOT	param->p[0]
#define R1		GSL_MAX(RTOT-DR,0.0) // parenthesised (2026-11-23): was RTOT-DR, i.e. q*RTOT-DR in q*R1

scalar sasfit_ff_linshell2(scalar q, sasfit_param * param)
{
	SASFIT_ASSERT_PTR(param);

	return gsl_pow_2(sasfit_ff_linshell2_f(q,param));
}

scalar sasfit_ff_linshell2_f(scalar q, sasfit_param * param)
{
	scalar m, b, Ro, eta_sh_in, eta_sh_out, shlin;
	scalar DR, eta_core, eta_shell, x_in_solvent, x_out_solvent, eta_solvent;

	SASFIT_ASSERT_PTR(param);

	sasfit_get_param(param, 7, EMPTY, &DR, &eta_core, &eta_shell, &x_in_solvent, &x_out_solvent, &eta_solvent);

	eta_sh_in  = (1.0- x_in_solvent)*eta_shell +  x_in_solvent*eta_solvent-eta_solvent;
	eta_sh_out = (1.0-x_out_solvent)*eta_shell + x_out_solvent*eta_solvent-eta_solvent;

	// stable for small q (2026-11-23): 4 pi int r^2 sinc(qr) dr = 4/3 pi R^3 phi(qR),
	// 4 pi int r^3 sinc(qr) dr = 4 pi R^4 chi(qR) with series for qR < 1 (the closed forms divided by q^3, q^4)
	shlin = fz_sphere(q, R1, eta_core-eta_solvent);

	if (DR > 0)
	{
		Ro = R1+DR;
		m = (eta_sh_out-eta_sh_in)/DR;
		b = -m*R1+eta_sh_in;
		shlin = shlin + fz_sphere(q, Ro, b) - fz_sphere(q, R1, b)
				+ 4.0*M_PI*m*(gsl_pow_4(Ro)*fz_chi(q*Ro) - gsl_pow_4(R1)*fz_chi(q*R1));
	}

	return shlin;
}

scalar sasfit_ff_linshell2_v(scalar q, sasfit_param * param, int dist)
{
	SASFIT_ASSERT_PTR(param); // assert pointer param is valid

	// insert your code here
	return 4./3.*M_PI*gsl_pow_3(fabs(RTOT)); // outer volume (2026-11-23), was 0
}

