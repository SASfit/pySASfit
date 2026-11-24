/*
 * Author(s) of this file:
 *   Joachim Kohlbrecher (joachim.kohlbrecher@psi.ch)
 */

// doi: 10.1063/5.0160446.
// M. Jura and M. Bishop,
// “The scattering functions of ideal tri-functional comb and dendrimer polymers,”
// The Journal of Chemical Physics,
// vol. 159, no. 3, Jul. 2023,

#include "include/private.h"
#include <sasfit_error_ff.h>
#include <math.h>


// define shortcuts for local parameters/variables
#define RG	    param->p[0]
#define DUMMY	param->p[1]
#define G       param->p[2]
#define I0	    param->p[3]


/*
 * 2026-11-24: the closed forms of generations 1-9 written as I = (2/u^2) sum_k c_k (exp(-l_k u) - 1 + l_k u)
 * = 2 sum_k c_k l_k^2 g(l_k u) (same coefficients c_k and decay rates l_k = num/den; the constant and linear
 * terms of the old expressions cancel exactly). The direct sums of exponentials (coefficients up to 786432)
 * lost digits for small u (2e-4 at u = 1e-3 for g = 9); fractional g was rounded with rint (extrapolating for
 * fractional parts > 0.5) and g = 9 read past the table.
 */
static const int dendrimer_n[10] = {1, 2, 3, 5, 9, 11, 13, 15, 17, 19};
static const scalar dendrimer_c[10][19][3] = {
    /* g = 0 (linear chain): Debye */ {{-1, 1, 1}},
    /* g = 1 */ {{-3, 1, 3}, {3, 2, 3}},
    /* g = 2 */ {{-3, 1, 9}, {-12, 1, 3}, {12, 4, 9}},
    /* g = 3 */ {{-3, 1, 21}, {6, 2, 21}, {-12, 1, 7}, {-48, 5, 21}, {48, 2, 7}},
    /* g = 4 */ {{-3, 1, 93}, {42, 2, 93}, {-12, 1, 31}, {72, 4, 93}, {-48, 5, 93}, {96, 2, 31}, {-192, 7, 93}, {-768, 3, 31}, {768, 10, 93}},
    /* g = 5 */ {{-3, 1, 189}, {90, 2, 189}, {-12, 1, 63}, {168, 4, 189}, {-48, 5, 189}, {288, 2, 63}, {-192, 1, 27}, {384, 8, 189}, {-768, 1, 21}, {-3072, 11, 189}, {3072, 4, 63}},
    /* g = 6 */ {{672, 2, 127}, {360, 4, 381}, {-12, 1, 127}, {-3, 1, 381}, {-3072, 11, 381}, {-12288, 13, 381}, {1152, 8, 381}, {-48, 5, 381}, {186, 2, 381}, {1536, 10, 381}, {-768, 3, 127}, {-192, 7, 381}, {12288, 14, 381}},
    /* g = 7 */ {{6144, 4, 255}, {-768, 1, 85}, {2688, 8, 765}, {-48, 1, 153}, {-12, 1, 255}, {378, 2, 765}, {-3, 1, 765}, {-192, 7, 765}, {-3072, 11, 765}, {-12288, 13, 765}, {-49152, 1, 51}, {1440, 2, 255}, {4608, 2, 153}, {744, 4, 765}, {49152, 16, 765}},
    /* g = 8 */ {{-196608, 17, 1533}, {24576, 2, 219}, {-12288, 13, 1533}, {10752, 10, 1533}, {5760, 8, 1533}, {-192, 1, 219}, {-48, 5, 1533}, {762, 2, 1533}, {-3, 1, 1533}, {-3072, 11, 1533}, {-49152, 5, 511}, {1512, 4, 1533}, {18432, 4, 511}, {-768, 3, 511}, {2976, 2, 511}, {-12, 1, 511}, {196608, 6, 511}},
    /* g = 9 */ {{-48, 5, 3069}, {-49152, 5, 1023}, {-12288, 13, 3069}, {43008, 4, 1023}, {23040, 10, 3069}, {-192, 7, 3069}, {6048, 2, 1023}, {3048, 4, 3069}, {-12, 1, 1023}, {-768, 1, 341}, {-3, 1, 3069}, {-3072, 1, 279}, {1530, 2, 3069}, {-196608, 17, 3069}, {-786432, 19, 3069}, {73728, 14, 3069}, {11904, 8, 3069}, {98304, 16, 3069}, {786432, 20, 3069}},
};

static scalar dendrimer_gen(scalar u, int gen)
{
	scalar s = 0.0, l;
	int k;
	if (gen == 0) return 2.0*hbp_g(u);
	for (k = 0; k < dendrimer_n[gen]; k++) {
		l = dendrimer_c[gen][k][1]/dendrimer_c[gen][k][2];
		s += dendrimer_c[gen][k][0]*l*l*hbp_g(l*u);
	}
	return 2.0*s;
}

scalar sasfit_ff_ideal_dendrimer(scalar q, sasfit_param * param)
{
    scalar X,u,w,res;
    int gl;

	SASFIT_ASSERT_PTR(param); // assert pointer param is valid

	SASFIT_CHECK_COND1((q < 0.0), param, "q(%lg) < 0",q);
	SASFIT_CHECK_COND1((RG <= 0.0), param, "Rg(%lg) <= 0",RG); // modify condition to your needs
	SASFIT_CHECK_COND1((G < 0.0), param, "g(%lg) < 0",G); // modify condition to your needs
	SASFIT_CHECK_COND1((G > 9.0), param, "g(%lg) > 9",G); // modify condition to your needs

    X= q*RG;
    u=X*X;
    gl = (int) floor(G);
    if (gl > 9) gl = 9;
    w = G-gl;
    res = dendrimer_gen(u, gl);
    if (w > 0.0) res = (1-w)*res + w*dendrimer_gen(u, gl+1);
    return I0*res;
}

scalar sasfit_ff_ideal_dendrimer_f(scalar q, sasfit_param * param)
{
	SASFIT_ASSERT_PTR(param); // assert pointer param is valid

	// insert your code here
	return 0.0;
}

scalar sasfit_ff_ideal_dendrimer_v(scalar q, sasfit_param * param, int dist)
{
	SASFIT_ASSERT_PTR(param); // assert pointer param is valid

	// insert your code here
	return 0.0;
}

