/*
 * src/plugins/triax_ellip_shell/sasfit_ff_triax_ellip_shell.c
 *
 * Copyright (c) 2008-2009, Paul Scherrer Institute (PSI)
 *
 * This file is part of SASfit.
 *
 * SASfit is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * SASfit is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with SASfit.  If not, see <http://www.gnu.org/licenses/>.
 */

/*
 * Author(s) of this file:
 *   Joachim Kohlbrecher (joachim.kohlbrecher@psi.ch)
 *   Ingo Bressler (ingo.bressler@bam.de)
 */

#include "include/private.h"
#include <sasfit_error_ff.h>

// define shortcuts for local parameters/variables

scalar triax_ellip_shell_cubature(const scalar *x, size_t ndim, void *pam) {
	sasfit_param subParam;
	sasfit_init_param( &subParam );
	sasfit_param *param;
	cubature_param *cparam;
	cparam = (cubature_param *) pam;
	param = cparam->param;

	if ((ndim < 2) ) {
		sasfit_err("false dimensions ndim:%d\n",ndim);
		return -1;
	}
	if ((ndim < 3) || (SIGMA==0)) {
		LNDISTR = 1;
		NU = 1;
	} else {
		subParam.p[0] = 1.0;
		subParam.p[1] = SIGMA;
		subParam.p[2] = 1.0;
		subParam.p[3] = 1.0;
		NU=x[2];
		LNDISTR = sasfit_sd_LogNorm(NU, &subParam);
		SASFIT_CHECK_SUB_ERR(param, subParam);
		if ( subParam.errStatus != FALSE ) {
			sasfit_err("LogNormError: SIGMA:%lf\n",SIGMA);
			return -1;
		}
	}
	X = x[0];
	Y = x[1];
    return (cparam->func)(param);
}

/*
 * Mean outer volume 4pi/3 <(a nu^ka + t nu^kt)(b nu^kb + t nu^kt)(c nu^kc + t nu^kt)> over the log-normal
 * distribution of nu (median 1, width sigma = p[4]): <nu^k> = exp(k^2 sigma^2/2). sa, sb, sc, st: 1 if the
 * length is scaled by nu. The distributed parameter p[dist] (dist = 0..4) is replaced by x. (2026-11-24: the
 * volume functions used the nu of the last cubature point, ignored x, and _t_v had the formula of _1.)
 */
scalar triax_ellip_shell_mean_volume(scalar x, sasfit_param *param, int dist, int sa, int sb, int sc, int st)
{
	scalar v[5], len[3], k[3], total = 0.0;
	int i, j, l, n;
	for (n = 0; n < 5; n++) v[n] = param->p[n];
	if (dist >= 0 && dist < 5) v[dist] = x;
	len[0] = v[0]; len[1] = v[1]; len[2] = v[2];
	k[0] = sa; k[1] = sb; k[2] = sc;
	for (i = 0; i < 2; i++)
		for (j = 0; j < 2; j++)
			for (l = 0; l < 2; l++) {
				scalar f = (i ? v[3] : len[0]) * (j ? v[3] : len[1]) * (l ? v[3] : len[2]);
				scalar kk = (i ? st : k[0]) + (j ? st : k[1]) + (l ? st : k[2]);
				total += f * exp(0.5 * v[4] * v[4] * kk * kk);
			}
	return 4.0 / 3.0 * M_PI * total;
}

/*
 * 3 (sin u - u cos u)/u^3 without cancellation for small u (series below 0.05; 2026-11-24: the inline closed form
 * lost digits for small q, e.g. 1e-8 at u = 1e-4, and stalled the adaptive integration of _syl at q = 1e-3).
 */
scalar triax_ellip_shell_phi(scalar u)
{
	scalar u2;
	if (fabs(u) < 0.05) {
		u2 = u*u;
		return 1.0 - u2/10.0*(1.0 - u2/28.0*(1.0 - u2/54.0*(1.0 - u2/88.0)));
	}
	return 3.0*(sin(u) - u*cos(u))/(u*u*u);
}
