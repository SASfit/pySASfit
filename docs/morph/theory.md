# pySASmorph: theory and mathematics

This page derives every step of {doc}`pySASmorph <index>` in the order the program runs. Each step names the
function that implements it. Lengths and q are in the units of the data, e.g. Å and Å⁻¹. Intensities are in the
units of the data, so absolute values need data on an absolute scale.

## 1. Two-phase structures and their scattering

A two-phase sample has the scattering length density (SLD)

$$
\eta(\mathbf r) = \rho_1 + \Delta\rho\, \chi(\mathbf r), \qquad
\chi(\mathbf r) = \begin{cases} 1 & \text{phase 2 ("pore")}\\ 0 & \text{phase 1 ("solid")} \end{cases}
$$

with the volume fraction $\varphi = \langle\chi\rangle$ of phase 2 and the contrast $\Delta\rho = \rho_2-\rho_1$.

**Two-point function.** The two-point probability $S_2(r) = \langle \chi(\mathbf 0)\chi(\mathbf r)\rangle$ of an
isotropic sample falls from $S_2(0) = \varphi$ to $S_2(\infty) = \varphi^2$. The normalised correlation function
of Debye and Bueche (1949) is

$$
\gamma(r) = \frac{\langle \delta\eta(\mathbf 0)\,\delta\eta(\mathbf r)\rangle}{\langle\delta\eta^2\rangle}
          = \frac{S_2(r)-\varphi^2}{\varphi(1-\varphi)}, \qquad \gamma(0) = 1 .
$$

**Scattering.** The macroscopic cross-section is its Fourier transform:

$$
I(q) = 4\pi\,\Delta\rho^2\,\varphi(1-\varphi) \int_0^\infty r^2\,\gamma(r)\,\frac{\sin qr}{qr}\,\mathrm dr .
$$

The inverse transform gives $\gamma$ from the data:

$$
\gamma(r) = \frac{1}{Q}\int_0^\infty q^2 I(q)\,\frac{\sin qr}{qr}\,\mathrm dq .
$$

Here $Q$ is the **invariant** (Porod 1951):

$$
Q = \int_0^\infty q^2 I(q)\,\mathrm dq = 2\pi^2\,\Delta\rho^2\,\varphi(1-\varphi) .
$$

**Sharp interfaces.** A sharp interface of area $S$ per volume $V$ gives Porod's law. It also fixes the slope of
$\gamma$ at the origin (Debye, Anderson and Brumberger 1957):

$$
I(q) \xrightarrow{q\to\infty} \frac{K}{q^4}, \quad K = 2\pi\,\Delta\rho^2\,\frac SV,
\qquad\qquad
\gamma(r) \simeq 1 - \frac{S/V}{4\varphi(1-\varphi)}\,r \quad (r\to 0) .
$$

Dividing $K$ by $Q$ removes the contrast:

$$
\frac SV = \pi\,\varphi(1-\varphi)\,\frac KQ
\qquad\text{({func}`~pysasfit.morph.statistics.specific_surface_porod`).}
$$

**Contrast from the invariant.** For a known $\varphi$ the invariant gives the contrast in the units of the data
({func}`~pysasfit.morph.scattering.contrast_from_invariant`):

$$
\Delta\rho^2 = Q / \bigl(2\pi^2\varphi(1-\varphi)\bigr) .
$$

All back-calculated curves use this contrast, so they are on the scale of the data. Conversely, with absolute
data ($I$ in cm⁻¹) and a known $\Delta\rho$ in cm⁻², $\varphi(1-\varphi) = Q/(2\pi^2\Delta\rho^2)$ gives the
volume fraction. The program takes the root $\le \tfrac12$
({func}`~pysasfit.morph.pipeline.porosity_from_invariant`). $Q$ is converted with 10²⁴ for q in Å⁻¹ and 10²¹ for
q in nm⁻¹.

## 2. Extending the data to q = 0 and q = ∞

The transforms need $I(q)$ on the whole q axis ({func}`~pysasfit.morph.extrapolation.extrapolate`).

**Background and Porod law.** The Porod law, with an optional flat background $B$, is fitted to the Porod range.
The fit is linear in $q^4$, by weighted least squares with the weights $w = 1/(\sigma q^4)$:

$$
I\,q^4 = K + B\,q^4 .
$$

$B$ is then subtracted from all data. $B$ can also be fixed by hand, and then it is subtracted before $K$ is
fitted. The parameter uncertainties are the least-squares covariance, scaled by $\chi^2_\text{red}$ when that is
larger than 1.

**The Porod range.** It is given by hand (the last N points, or $q \ge q_\text{min}$) or chosen automatically
({func}`~pysasfit.morph.extrapolation.auto_porod_range`); the former default was the last 15 %, at least 5 points.
The automatic choice works in three steps:

1. **Candidates.** Every start $q_i$ from 40 % of the points on (up to 60 candidates, at least 6 points) is fitted.
   With $\varepsilon_j = (I_j - Kq_j^{-4} - B)/\sigma_j$ (relative residuals without errors) it gets

   $$
   \chi^2_i = \frac{1}{N_i-p}\sum_{j\ge i}\varepsilon_j^2, \qquad
   \rho_i = \frac{\sum_j \delta_j\,\delta_{j+1}}{\sum_j\delta_j^2},\quad \delta_j = \varepsilon_j - \bar\varepsilon .
   $$

   $\rho_i$ is the lag-1 autocorrelation of the residuals.
2. **Acceptance.** A start is acceptable when two conditions hold:
   - $\chi^2_i \le 1.5\,\min_k\chi^2_k$, or $\chi^2_i \le 2$ with error bars;
   - $\rho_i < \max(0.15,\ 3/\sqrt{N_i})$, i.e. no trend beyond 3σ of white noise.

   A range reaching into the non-Porod region leaves residuals of one sign even when generous error bars keep
   $\chi^2$ small. The lowest acceptable $q_i$ gives the longest range and so the most precise $K$ and $B$; if no
   start is acceptable, the best fit is taken.
3. **No clear Porod regime.** The report warns in three cases:
   - the best $\chi^2 > 6$ with error bars;
   - the range spans less than a factor 1.5 in q, so $K$ and $B$ cannot be separated;
   - $Kq_i^{-4} < |B|$ at the start, so the "fit" is a constant.

A fitted $B$ also absorbs part of the next term of the Porod expansion (e.g. $\propto q^{-6}$) near the start of
the range.

**Low q.** The first points (default 10) give the extrapolation from q = 0 to $q_\text{min}$, on 40 nodes
({func}`~pysasfit.morph.extrapolation.fit_low_q`).

| method | model | fit |
|---|---|---|
| Guinier | $I_0\,e^{-q^2R_g^2/3}$ | $\ln I = a + b\,q^2$, weights $I/\sigma$; $R_g^2 = -3b$ |
| linear | $a + b\,q$ | weights $1/\sigma$ |
| constant | $\bar I = \sum I\sigma^{-2} / \sum\sigma^{-2}$ | – |

Guinier falls back to constant when $b \ge 0$ or some $I \le 0$. Linear falls back to constant when $a \le 0$.

**Between data points.** The program inserts 4 extra nodes per interval, with $I$ interpolated linearly in
$\log I$ against $\log q$ (linearly where $I \le 0$). On the resulting nodes $q_0 < \dots < q_N$ the function

$$
u(q) = q\,I(q)
$$

is treated as **piecewise linear**. Beyond $q_N$ the Porod law $K q^{-4}$ holds up to infinity.

**Invariant.** The invariant is exact for this interpolant, with the Porod tail integrated analytically
({attr}`~pysasfit.morph.extrapolation.ExtrapolatedCurve.invariant`):

$$
Q = \sum_i \int_{q_i}^{q_{i+1}} q\,u(q)\,\mathrm dq + \frac{K}{q_N} .
$$

The program reports the share $K/(q_N Q)$ of the Porod tail. Above 20 % the q range is probably too short.

## 3. Exact sine transforms

Every isotropic transform in pySASmorph has the form $T(r) = \int_{x_0}^{x_N} u(x)\sin(xr)\,\mathrm dx$ with $u$
piecewise linear ({func}`~pysasfit.morph.transforms.sine_transform_pl`). For example, $Q\gamma(r) = T(r)/r$ with
$u = qI$.

**Exact formula.** On a segment $[a,b]$ with slope $s$:

$$
\int_a^b u(x)\sin(xr)\,\mathrm dx
= \frac{u_a\cos ar - u_b\cos br}{r} + s\,\frac{\sin br - \sin ar}{r^2} .
$$

Summed over the segments, the first term telescopes to $\bigl(u_0\cos x_0 r - u_N\cos x_N r\bigr)/r$. This
Filon-type quadrature is exact for the interpolant at any $r$. It needs no points per oscillation. A trapezoid
rule on the data grid breaks down once $r\,\Delta q \sim 1$; that is where the ripple of SAXSMorph's $\Gamma(r)$
and $f(k)$ comes from.

**Small r.** For $r\,x_N < 0.05$ the closed form loses digits by cancellation. The program then uses the Taylor
series with the exact moments $m_n = \int x^n u\,\mathrm dx$ of the interpolant
({func}`~pysasfit.morph.transforms.moments_pl`):

$$
T(r) = r\,m_1 - \frac{r^3 m_3}{3!} + \frac{r^5 m_5}{5!} - \frac{r^7 m_7}{7!} .
$$

**Porod tail.** The tail $I = Kq^{-4}$ for $q>q_N$ contributes, with the sine integral $\mathrm{Si}$
({func}`~pysasfit.morph.transforms.porod_tail`):

$$
\int_{q_N}^\infty q^2\,\frac{K}{q^4}\,\frac{\sin qr}{qr}\,\mathrm dq = K\,r\,J_3(q_N r),
\qquad
J_3(a) = \int_a^\infty \frac{\sin x}{x^3}\,\mathrm dx
       = \frac{\sin a}{2a^2} + \frac{\cos a}{2a} - \frac12\Bigl(\frac\pi2 - \mathrm{Si}(a)\Bigr) .
$$

Its limit at $r = 0$ is $K/q_N$, the Porod part of the invariant.

## 4. The clipped Gaussian random field

**Gaussian random field.** $\psi(\mathbf r)$ is a statistically homogeneous and isotropic Gaussian random field
with zero mean, unit variance and correlation function

$$
g(r) = \langle\psi(\mathbf 0)\,\psi(\mathbf r)\rangle, \qquad g(0) = 1 .
$$

**Level cut.** Cutting it at the level $\alpha$ gives a two-phase structure (Cahn 1965; Berk 1987, 1991):

$$
\chi(\mathbf r) = H\bigl(\psi(\mathbf r)-\alpha\bigr), \qquad
\varphi = P(\psi>\alpha) = \tfrac12\operatorname{erfc}\bigl(\alpha/\sqrt2\bigr),\quad
\alpha = \sqrt2\,\operatorname{erfc}^{-1}(2\varphi) .
$$

**Two-point function of the cut field.** $\psi(\mathbf 0)$ and $\psi(\mathbf r)$ are bivariate normal with
correlation $g$. With the bivariate normal density $\phi_2$, Plackett's identity (1954) states

$$
\frac{\partial}{\partial t}\,P\bigl(\psi_1>a,\ \psi_2>b\bigr)\Big|_{\operatorname{corr}=t} = \phi_2(a,b;t),
\qquad
\phi_2(a,b;t) = \frac{1}{2\pi\sqrt{1-t^2}}\exp\!\Bigl[-\frac{a^2+b^2-2tab}{2(1-t^2)}\Bigr] .
$$

At $t = 0$ the two values are independent and $S_2 = \varphi^2$. Integrating from there gives

$$
S_2(r)-\varphi^2 = F\bigl(g(r)\bigr),\qquad
F(g) = \frac{1}{2\pi}\int_0^{g}\frac{e^{-\alpha^2/(1+t)}}{\sqrt{1-t^2}}\,\mathrm dt
     = \frac{1}{2\pi}\int_0^{\arcsin g} e^{-\alpha^2/(1+\sin\theta)}\,\mathrm d\theta .
$$

**Implementation.** $F$ is tabulated in $\theta = \arcsin g$ on 20 001 points, where the integrand has no end-point
singularity, and integrated by cumulative Simpson ({class}`~pysasfit.morph.correlation.LevelCut`).

**End values.** They are the exact orthant probabilities. At $g = 1$ the two points coincide. At $g = -1$
$\psi(\mathbf r) = -\psi(\mathbf 0)$, so both values exceed $\alpha$ only between $\alpha$ and $-\alpha$:

$$
F(1) = \varphi(1-\varphi), \qquad F(-1) = -\min(\varphi,1-\varphi)^2,
\qquad F(g) = \frac{\arcsin g}{2\pi}\ \ (\varphi=\tfrac12).
$$

**Inversion.** Setting the model equal to the data, $S_2-\varphi^2 = \varphi(1-\varphi)\gamma$, gives the field
correlation:

$$
g(r) = F^{-1}\bigl(\varphi(1-\varphi)\,\gamma(r)\bigr) .
$$

$F$ is monotonic. It is inverted by interpolation in the table (θ as a function of F), then $g = \sin\theta$.
Values of $\varphi(1-\varphi)\gamma$ outside $[F(-1), F(1)]$ cannot come from any level-cut field. They are
clipped and counted ({attr}`~pysasfit.morph.correlation.LevelCut.clipped`). Many clipped points mean a wrong
$\varphi$, a wrong background, or a structure that is not a level cut.

**Interfaces of the cut field.** Near $g = 1$ the integrand is $\simeq e^{-\alpha^2/2}/\sqrt{2(1-t)}$, so

$$
F(1)-F(g) \simeq \frac{e^{-\alpha^2/2}}{2\pi}\sqrt{2(1-g)} .
$$

A smooth field has $g \simeq 1 + \tfrac12 g''(0)\,r^2$. Then $1-\gamma$ is linear in $r$, which is Porod
behaviour. Comparison with §1 gives the interface area of the level surfaces:

$$
\frac SV = \frac2\pi\,\sqrt{-g''(0)}\;e^{-\alpha^2/2} .
$$

This is twice Rice's (1944) number of level crossings per unit length; the factor 2 is the stereological
relation $S/V = 2P_L$. Data that fall off more slowly than $q^{-4}$ (rough or fractal interfaces) need a
non-smooth $g$ and lie outside the model.

## 5. Spectral function

The field is built from its spectral function, the 3D Fourier transform of $g$
({func}`~pysasfit.morph.correlation.spectral_function`):

$$
f(k) = 4\pi\int_0^\infty r^2\,g(r)\,\frac{\sin kr}{kr}\,\mathrm dr,
\qquad
g(r) = \frac{1}{2\pi^2}\int_0^\infty k^2 f(k)\,\frac{\sin kr}{kr}\,\mathrm dk .
$$

It is computed with the transform of §3, with $u = r\,g$.

**Wave-vector distribution.** Its distribution of magnitudes is normalised by $g(0) = 1$
({func}`~pysasfit.morph.correlation.wavevector_distribution`):

$$
P(k) = \frac{k^2 f(k)}{2\pi^2}, \qquad \int_0^\infty P(k)\,\mathrm dk = 1 .
$$

Negative values of $P$ are set to zero. A valid correlation function has $f \ge 0$ (Bochner), so negative values
signal an inconsistent $g$.

**r grid.** $g$ is evaluated on a uniform grid (`pipeline._r_grid`):
- **Range:** from 0 to $r_\text{max} = 1.02\cdot\tfrac{\sqrt3}{2}L$, half the box diagonal.
- **Step:** $\Delta r = \min\bigl(a/4,\ \pi/(16\,q_\text{max})\bigr)$, so that the model $I(q)$ is accurate up to
  $q_\text{max}$.

A warning is given when $|\gamma(r_\text{max})| > 0.05$, i.e. when the correlations extend beyond the box.

## 6. Generating the field

The box has edge $L$ with $n^3$ voxels of size $a = L/n$ ({mod}`pysasfit.morph.grf`).

### FFT spectral synthesis (default)

{func}`~pysasfit.morph.grf.field_fft` follows the circulant-embedding idea (Dietrich and Newsam 1997):

1. **Sample $g$ on the grid.** $g$ is sampled at the minimum-image distance of every voxel to the origin. This
   periodises $g$ on the torus of edge $L$; beyond $r_\text{max}$, $g = 0$.
2. **Power spectrum.** $S(\mathbf k) = \operatorname{Re}\,\mathrm{DFT}[g]$ on exactly the grid wave vectors
   $\mathbf k = 2\pi\mathbf m/L$. Negative values, which come from a truncated or inconsistent $g$, are set to
   zero. Their share of $\sum|S|$ is reported as `negative_power`.
3. **Field.** With white Gaussian noise $w$ on the grid:

   $$
   \psi = \mathrm{IFFT}\bigl[\sqrt{S}\;\mathrm{FFT}(w)\bigr],
   $$

   then shifted to mean 0 and scaled to variance 1.

The covariance of $\psi$ on the grid is then exactly the periodised $g$, apart from the discarded negative power.
The cost is $O(n^3\log n)$. The field is periodic, so $L$ must span several correlation lengths.

### Random waves (Berk; SAXSMorph)

{func}`~pysasfit.morph.grf.field_cosine`, with `method="cosine"`:

$$
\psi(\mathbf r) = \sqrt{\frac{2}{N_w}}\sum_{i=1}^{N_w}\cos(\mathbf k_i\cdot\mathbf r + \phi_i) .
$$

- $|\mathbf k_i|$ is drawn from $P(k)$ by inverse CDF. SAXSMorph uses rejection sampling.
- The directions are uniform on the sphere and the phases $\phi_i$ uniform in $[0, 2\pi)$.

For $N_w\to\infty$ the field is Gaussian with correlation $g$. It is not periodic. The cost is
$O(N_w n^3)$. The share of $|\mathbf k_i| > \pi/a$, which the grid cannot represent, is reported.

### The cut

{func}`~pysasfit.morph.grf.level_cut` has two modes:
- **Quantile (default):** the cut is at the empirical $(1-\varphi)$-quantile of the voxel values, so the voxel
  porosity equals $\varphi$ exactly.
- **α:** the cut is at the theoretical level $\alpha$, so the voxel porosity fluctuates around $\varphi$.

## 7. Checks against the data

**Level-cut model** ({func}`~pysasfit.morph.scattering.levelcut_intensity`). This is the ensemble average of the
model with the reconstructed $g$. It equals the data wherever $\gamma$ could be inverted without clipping:

$$
I_\text{lc}(q) = \Delta\rho^2\,4\pi\int_0^{r_\text{max}} r^2\,F\bigl(g(r)\bigr)\,\frac{\sin qr}{qr}\,\mathrm dr .
$$

With measurement errors the report gives
$\chi^2 = \frac1N\sum\bigl[(I_\text{lc}-I)/\sigma\bigr]^2$ over the data points used (mean square, not divided by
degrees of freedom).

**Voxel structure** ({func}`~pysasfit.morph.scattering.voxel_intensity`). This is what one realisation scatters:

$$
I_\text{vox}(q) = \Delta\rho^2\,\frac{a^3}{n^3}\,\Bigl\langle\bigl|\mathrm{DFT}[\chi-\bar\chi]\bigr|^2\Bigr\rangle_{|\mathbf k|\approx q} .
$$

It is averaged in shells of width $2\pi/L$; the half-space rfft planes are weighted twice. Expect speckle at
low q, from the few wave vectors per shell, and voxel effects near $\pi/a$.

## 8. Morphology of the voxel structure

{mod}`pysasfit.morph.statistics` and {func}`~pysasfit.morph.multiphase.interface_areas` compute:

- **Interface area.**

  $$
  \frac SV = \frac23\,\frac{N_\text{faces}\,a^2}{n^3a^3},
  $$

  with $N_\text{faces}$ the voxel faces between unlike neighbours (periodic box). A staircase surface overestimates
  the area by the mean of $|n_x|+|n_y|+|n_z|$ over isotropic normals, which is $3/2$; hence the factor $2/3$.
  Compare with $S/V$ from Porod (§1).
- **Clusters.** Connected components of each phase with 6-connectivity, the largest cluster's share, and their
  number.
- **Percolation** along x, y and z: a cluster touches both opposite faces of the box.
- **Bicontinuous** (as SAXSMorph): each phase has a cluster that percolates in all three directions.
- **Degree of connectivity** (as SAXSMorph): $(N_\text{largest pore}+N_\text{largest solid})/n^3$.

## 9. Several phases: one field, several cuts

### Ordered phases

{mod}`pysasfit.morph.multiphase`. The same field is cut at $N-1$ levels $\alpha_1<\dots<\alpha_{N-1}$. Phase $i$
occupies $\alpha_i<\psi<\alpha_{i+1}$, with $\alpha_0 = -\infty$ and $\alpha_N = \infty$. The phases are therefore
*ordered*: phase $i$ touches only $i\pm1$. A middle phase is a layer between its neighbours, e.g. a film, a
shell or a wetting layer. The fractions fix the levels, with $\Phi$ the standard normal distribution function:

$$
\alpha_k = \Phi^{-1}\Bigl(\sum_{i<k}\varphi_i\Bigr) .
$$

### SLD correlation

With the SLD jumps $D_k = \rho_k-\rho_{k-1}$ the SLD is

$$
\rho(\psi) = \rho_0 + \sum_k D_k\,H(\psi-\alpha_k) .
$$

Plackett's identity (§4) applied to every pair of cuts gives the SLD correlation:

$$
C(r) = \langle\delta\rho(\mathbf 0)\,\delta\rho(\mathbf r)\rangle
     = \sum_{k,l} D_k D_l\, B\bigl(\alpha_k,\alpha_l;g(r)\bigr),\qquad
B(a,b;g) = \int_0^g \phi_2(a,b;t)\,\mathrm dt .
$$

It starts from

$$
C(0) = \sum_i\varphi_i\rho_i^2 - \Bigl(\sum_i\varphi_i\rho_i\Bigr)^2 .
$$

- **Special cases:** one cut reduces to §4; a single phase between two cuts is the two-cut model of Berk (1991),
  Roberts (1997) and Arleth et al. (2001).
- **Implementation:** $B$ is tabulated in $\theta = \arcsin t$, like $F$.
- **Invariant:** $Q_m = s\cdot 2\pi^2 C_m(0)$. The unit factor $s$ is $10^{-4}$ (q in Å⁻¹) or $10^{-1}$ (q in
  nm⁻¹), for SLDs in 10⁻⁶ Å⁻² and $I$ in cm⁻¹.

### Several contrasts

Curve $m$ has its own SLDs and gives

$$
\gamma_m(r) = C_m(r)/C_m(0) \quad\text{(data)},\qquad G_m(g) = C_m(g)/C_m(0) \quad\text{(model)} .
$$

**Common field correlation.** At every $r$ it minimises the weighted misfit
({func}`~pysasfit.morph.multiphase.joint_g`):

$$
g(r) = \arg\min_g \sum_m w_m\bigl[\gamma_m(r) - G_m(g)\bigr]^2 .
$$

The minimum is searched on a fine $g$ grid, so it works where $G_m$ is not monotonic.

**Rescaling.** For two or more curves each $\gamma_m$ may be rescaled by $\lambda_m$. This allows for an invariant
that is a few per cent off when a curve does not reach its Porod regime:

$$
\lambda_m = \frac{\sum_r G_m(g)\,\gamma_m}{\sum_r\gamma_m^2} .
$$

$\lambda_m$ is clipped to $[0.8, 1.25]$, and $g$ and $\lambda$ are iterated. Corrections above 10 % are reported.

**Fitting the fractions** ({func}`~pysasfit.morph.multiphase.fit_fractions`):
- **Parametrisation:** $\varphi_i = e^{x_i}/\sum e^{x_j}$ with $x_0 = 0$, minimised by Nelder–Mead.
- **Objective:** the profile misfit $\sum_m w_m\,\overline{(\gamma_m - G_m)^2}$, plus $10^{-2}\sum_m w_m
  \bigl[\ln\frac{Q_m}{2\pi^2C_m(0)} - c\bigr]^2$.
- **Invariant offset $c$:** the weighted mean of the logarithms when all curves share one intensity scale (only
  the ratios count), or $\ln s$ for absolute data.
- **Weight:** with $10^{-2}$, a 10 % invariant mismatch counts as an rms $\gamma$ misfit of 0.01.
- **Accuracy:** the shapes alone fix the fractions only weakly, to a few hundredths in the tests.

**Interfaces between phases.** $S_{ij}/V$ is found by the face count of §8, for every pair of phases $i<j$.
Non-neighbouring phases touch only where a layer is thinner than a voxel.

## 10. Surfaces

The 3D panel and the STL export draw iso-surfaces by marching cubes (Lorensen and Cline 1987;
{mod}`pysasfit.morph.surface`). Three sources are possible:
- **Exact cut:** the field $\psi$ at the cut level, i.e. the smooth interface of the model. For a middle phase it
  is the pair of level surfaces.
- **Smoothed voxels:** the voxel matrix smoothed with a Gaussian of 0.8 voxel.
- **Voxels as exported.**

The volume is padded with one layer of the other phase, so the surface is closed at the box faces and at the
clipping plane.

## 11. Box and grid

- **Box size:** by default $L = 5\cdot 2\pi/q^*$, with $q^*$ the maximum of the Kratky plot $q^2I$, but at most
  $n\cdot 2\pi/q_\text{max}$ ({func}`~pysasfit.morph.pipeline.suggest_box`).
- **Voxel size:** the finest structure the data resolve is about $\pi/q_\text{max}$. A warning is given when
  $\pi/a < q_\text{max}$.
- **Correlations:** $\gamma(r)$ must have decayed within half the box diagonal (warning, §5).

## 12. Uniaxial anisotropy (texture)

{mod}`pysasfit.morph.anisotropic`; usage on {doc}`texture`.

### The level cut holds direction by direction

Nothing in §4 needs isotropy. $\psi(\mathbf 0)$ and $\psi(\mathbf r)$ are bivariate normal with correlation
$g(\mathbf r)$ for every vector $\mathbf r$, so

$$
S_2(\mathbf r)-\varphi^2 = F\bigl(g(\mathbf r)\bigr), \qquad g(\mathbf r) = F^{-1}\bigl(\varphi(1-\varphi)\,\gamma(\mathbf r)\bigr) .
$$

The FFT synthesis of §6 works with any $g(\mathbf r)$ sampled on the grid
({func}`~pysasfit.morph.anisotropic.field_from_grid_correlation`). What changes is how $\gamma(\mathbf r)$ is
obtained from the data.

### Uniaxial symmetry: one 2D pattern is enough

Take a sample symmetric about an axis $\hat{\mathbf e}$ in the detector plane, at the angle $\psi$ from $q_x$. Its
intensity depends only on $q$ and

$$
\mu = |\hat{\mathbf q}\cdot\hat{\mathbf e}| ,
$$

since centrosymmetry folds $\mu$ to $[0,1]$. At small angles the detector plane ($q_z = 0$) contains $\hat{\mathbf
e}$, so its points have every $\mu$, and the 2D pattern determines $I(q,\mu)$ completely. Along and across the
axis:

$$
q_\parallel = q_x\cos\psi + q_y\sin\psi, \qquad q_\perp = -q_x\sin\psi + q_y\cos\psi .
$$

### Model B: Legendre components and the plane-wave expansion

**Components.** The intensity is expanded in even Legendre polynomials:

$$
I(q,\mu) = \sum_{l=0,2,\dots}^{l_\text{max}} I_l(q)\,P_l(\mu) .
$$

The $I_l$ are found by weighted least squares in q bins
({func}`~pysasfit.morph.anisotropic.legendre_components`). Two details matter:

- **Bin flattening:** each pixel is first moved to the bin's mean q along the azimuthal mean
  $I_\text{ref}(q)$. Otherwise the correlation of q and μ inside a bin on a square pixel grid appears as a
  spurious anisotropy.
- **Coverage:** a bin uses only as many orders as its μ coverage supports (occupied tenths of [0, 1], minus one).
  Bins without full coverage are left out.

**Correlation components.** With the plane-wave expansion
$e^{i\mathbf q\cdot\mathbf r} = \sum_l (2l+1)\,i^l j_l(qr)\,P_l(\hat{\mathbf q}\cdot\hat{\mathbf r})$ and the
addition theorem, $\int P_l(\hat{\mathbf q}\cdot\hat{\mathbf e})\,P_{l'}(\hat{\mathbf q}\cdot\hat{\mathbf r})\,
\mathrm d\Omega_q = \frac{4\pi}{2l+1}\delta_{ll'}P_l(\hat{\mathbf r}\cdot\hat{\mathbf e})$:

$$
C(\mathbf r) = \frac{1}{(2\pi)^3}\int I(\mathbf q)\,e^{i\mathbf q\cdot\mathbf r}\,\mathrm d^3q
            = \sum_l C_l(r)\,P_l(\hat{\mathbf r}\cdot\hat{\mathbf e}),
\qquad
C_l(r) = \frac{(-1)^{l/2}}{2\pi^2}\int_0^\infty q^2 I_l(q)\,j_l(qr)\,\mathrm dq .
$$

Then $\gamma(\mathbf r) = C(\mathbf r)/C_0(0)$.

**The isotropic part.** Only $l = 0$ contributes to the invariant:

$$
\int I\,\mathrm d^3q = 4\pi Q_0, \qquad Q_0 = \int q^2 I_0\,\mathrm dq = 2\pi^2\Delta\rho^2\varphi(1-\varphi),
$$

so the contrast and $S/V$ (from $K_0$) follow as in §1. $C_0$ is the transform of §3, with the flat background
B, the low-q extrapolation and the Porod tail all applied to $I_0$.

**Higher orders, $l \ge 2$.** $I_l$ is extended in three ranges:
- below the data as $I_l(q_1)(q/q_1)^2$ (the anisotropy vanishes at $q = 0$);
- within the data, linearly;
- beyond the data as $K_l q^{-4}$, with $K_l$ fitted on the Porod range.

The finite part is integrated on a fine uniform grid. The tail is analytic:

$$
\int_{q_N}^\infty q^2\,\frac{K_l}{q^4}\,j_l(qr)\,\mathrm dq = K_l\,r\int_{q_Nr}^\infty\frac{j_l(x)}{x^2}\,\mathrm dx,
\qquad
\int_0^\infty\frac{j_l(x)}{x^2}\,\mathrm dx = \frac{\sqrt\pi}{8}\,\frac{\Gamma\bigl(\frac{l-1}{2}\bigr)}{\Gamma\bigl(\frac{l+4}{2}\bigr)}
\quad (l\ge2),
$$

e.g. π/16 for l = 2 ({func}`~pysasfit.morph.anisotropic.spherical_transform`).

### Model A: affine stretch

An isotropic structure stretched by $\lambda$ along the axis, $\eta(\mathbf r) = \eta_\text{iso}(M^{-1}\mathbf r)$
with $M = \operatorname{diag}(\lambda, 1, 1)$, has

$$
\gamma(\mathbf r) = \gamma_\text{iso}\Bigl(\sqrt{r_\parallel^2/\lambda^2 + r_\perp^2}\Bigr),
\qquad
I(\mathbf q) = \lambda\,I_\text{iso}\Bigl(\sqrt{\lambda^2 q_\parallel^2 + q_\perp^2}\Bigr) .
$$

**Fitting λ and the axis** ({func}`~pysasfit.morph.anisotropic.fit_affine`). Both are fitted by collapsing
$\log I$ onto one curve of $q_\text{eff}$. The misfit is the weighted scatter of $\log I$ about its mean in
quantile bins of $q_\text{eff}$:
- first a grid search over λ ∈ [0.2, 5];
- then Nelder–Mead in (ln λ, ψ).

The collapsed curve $I/\lambda$ against $q_\text{eff}$ goes through the isotropic pipeline. Its invariant equals
$Q_0$, because $\int I\,\mathrm d^3q$ is invariant under the stretch.

**Which axis.** In the plane, λ along ψ and 1/λ along ψ + 90° give the same ellipses. They differ in 3D, because
the structure is rotationally symmetric about the chosen axis. The choice of axis therefore decides the 3D
structure.

### Field, model and checks

- **g on the grid.** $g(r,\mu)$ is tabulated on the r grid of §5 and 65 values of μ. The grid vectors take their
  value by bilinear interpolation. The axis is the box x axis.
- **Level-cut model.** $c(r,\mu) = F(g(r,\mu))$ is projected onto Legendre polynomials by 64-point
  Gauss–Legendre quadrature in μ. A uniform rule fails here, because $c$ has a cusp at r = 0. Then

  $$
  I^\text{lc}_l(q) = \Delta\rho^2\,4\pi\,(-1)^{l/2}\int_0^{r_\text{max}} r^2\,c_l(r)\,j_l(qr)\,\mathrm dr,
  \qquad c_l(r) = (2l+1)\int_0^1 c(r,\mu)\,P_l(\mu)\,\mathrm d\mu ,
  $$

  up to $l = 12$. χ² is taken over all pixels in the q range.
- **Voxel structure.** The components come from all 3D wave vectors of $|\mathrm{DFT}[\chi-\bar\chi]|^2$, with
  μ = |k_x|/k.
- **Texture measures** ({func}`~pysasfit.morph.anisotropic.chord_lengths`):
  - the share of the interface faces normal to x, y and z (1/3 each when isotropic);
  - the mean chord length of each phase along each axis, $\varphi\,n^3a/(N_d/2)$, where $N_d$ is the number of
    faces normal to axis $d$.

## References

**Scattering by two-phase media**
* P. Debye, A. M. Bueche, *J. Appl. Phys.* **20** (1949) 518 – correlation function and scattering.
* P. Debye, H. R. Anderson, H. Brumberger, *J. Appl. Phys.* **28** (1957) 679 – random two-phase media, S/V from
  the slope of γ at the origin.
* G. Porod, *Kolloid-Z.* **124** (1951) 83; **125** (1952) 51 – the invariant and the q⁻⁴ law.
* A. Guinier, G. Fournet, *Small-Angle Scattering of X-Rays*, Wiley (1955).
* O. Glatter, O. Kratky (eds.), *Small Angle X-ray Scattering*, Academic Press (1982).

**Gaussian random fields and level crossings**
* S. O. Rice, *Bell Syst. Tech. J.* **23** (1944) 282; **24** (1945) 46 – level crossings of Gaussian processes.
* R. L. Plackett, *Biometrika* **41** (1954) 351 – derivative of the bivariate normal distribution with respect to
  the correlation.
* J. W. Cahn, *J. Chem. Phys.* **42** (1965) 93 – level surfaces of Gaussian waves as models of spinodal
  structures.
* R. J. Adler, *The Geometry of Random Fields*, Wiley (1981).

**Level-cut (clipped) Gaussian field models**
* N. F. Berk, *Phys. Rev. Lett.* **58** (1987) 2718; *Phys. Rev. A* **44** (1991) 5069 – scattering of level-cut
  fields, random-wave construction, two-cut (boundary-layer) model.
* M. Teubner, *Europhys. Lett.* **14** (1991) 403 – level surfaces of Gaussian fields and microemulsions.
* A. P. Roberts, M. Teubner, *Phys. Rev. E* **51** (1995) 4141 – properties of materials modelled by Gaussian
  random fields.
* A. P. Roberts, M. A. Knackstedt, *Phys. Rev. E* **54** (1996) 2313.
* A. P. Roberts, *Phys. Rev. E* **55** (1997) R1286 – two-cut and intersection models and their two-point functions.
* J. A. Quintanilla, *Modelling Simul. Mater. Sci. Eng.* **15** (2007) S337.

**Reconstruction from two-point statistics**
* B. Ingham, H. Li, E. L. Allen, M. F. Toney, *J. Appl. Cryst.* **44** (2011) 221, doi:10.1107/S0021889810048557 –
  SAXSMorph.
* C. L. Y. Yeong, S. Torquato, *Phys. Rev. E* **57** (1998) 495 – reconstructing random media.
* S. Torquato, *Random Heterogeneous Materials*, Springer (2002).

**Several phases and contrasts**
* L. Arleth, S. Marčelja, T. Zemb, *J. Chem. Phys.* **115** (2001) 3923 – two level cuts for asymmetric
  microemulsions (surfactant film), SANS.
* C. J. Gommes, A. P. Roberts, *Phys. Rev. E* **77** (2008) 041409; C. J. Gommes, *J. Appl. Cryst.* **46** (2013)
  493 – plurigaussian three-phase models, in situ SAS; C. J. Gommes, *Microporous Mesoporous Mater.* **257** (2018)
  62 – review of stochastic models.
* S. Kronenberger et al., *Soft Matter* (2024), doi:10.1039/d4sm00721b – three-phase polymer structures from two
  independent fields, anisotropic 2D data.

**Anisotropy (§12)**
* The plane-wave (Rayleigh) expansion and the addition theorem of spherical harmonics: e.g. G. B. Arfken,
  H. J. Weber, F. E. Harris, *Mathematical Methods for Physicists*, Academic Press.
* The closed form of $\int_0^\infty x^\mu j_l(x)\,\mathrm dx$: I. S. Gradshteyn, I. M. Ryzhik, *Table of
  Integrals, Series, and Products*, 6.561.

**Numerical methods**
* L. N. G. Filon, *Proc. R. Soc. Edinburgh* **49** (1928) 38 – quadrature of oscillatory integrals.
* C. R. Dietrich, G. N. Newsam, *SIAM J. Sci. Comput.* **18** (1997) 1088 – circulant embedding for Gaussian
  random fields.
* W. E. Lorensen, H. E. Cline, *Computer Graphics (SIGGRAPH '87)* **21** (1987) 163 – marching cubes.
