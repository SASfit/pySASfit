# SESANS and multiple scattering

pySASstudio has the **transform** of SASfit: the model can be shown and fitted as the scattering intensity, as a
SESANS signal, or as an intensity with multiple small-angle scattering. The choice is in the **Transform** box of the
Model panel, below the mixture. It corresponds to SASfit's *transform* menu:

| SASfit             | Transform box                         | what is computed                                                     |
| ------------------ | ------------------------------------- | -------------------------------------------------------------------- |
| `1`                | `1   I(Q)`                            | the intensity I(Q)                                                   |
| `H{…}/2pi`         | `H{…}/2π   SESANS  G(δ)−G(0)`         | G(δ) − G(0) = 1/(2π) ∫ I(Q) [J0(Qδ) − 1] Q dQ = ln(P/P0)/(λ²t)       |
| `Exp(H{…}/2pi)`    | `Exp(λ²t·H{…}/2π)   SESANS  P/P0`     | P/P0 = exp(λ²t [G(δ) − G(0)]) (λ = t = 1 gives SASfit's exp(G−G0))   |
| `MSAS`             | `MSAS   multiple small-angle scattering` | I(Q) with multiple scattering in a slab of thickness t            |

The SESANS transforms change the x axis from q to the **spin-echo length δ**. With MSAS the axis stays q, and the
single-scattering curve is drawn dash-dotted next to the curve with multiple scattering.

## Units

SASfit functions carry no units, so keep them consistent, as in SASfit:

- the intensity in absolute units, **1/cm**;
- the sample **thickness t in cm**;
- the **wavelength λ in the reciprocal unit of Q**: nm for Q in 1/nm, Å for Q in 1/Å.

G(δ) − G(0) then has the unit cm⁻¹·[Q]²: Å⁻²cm⁻¹ for Q in 1/Å, which is the unit of the "Depolarisation" column of
`.ses` files. λ²t·G is dimensionless, and λ²t·G(0) = Σt is the probability that a neutron is scattered (once or more)
in the sample. The q unit selector of the plot decides the length unit: 1/nm → δ and λ in nm, 1/Å → Å.

## SESANS data

**Add data…** in the Fit panel reads `.ses` files (and other text files whose header mentions the spin-echo length).
A dialog shows what was found in the file and asks for:

- the units of the spin-echo length and the wavelength *in the file*. Some files print a wrong unit (the
  core-shell example in sasfit-collab writes "µm" for values that are nm); choose the right one here;
- the sample thickness in cm (read from the header if it is there, also from `Thickness [mm]`);
- what to fit and show: G(δ) − G(0) or P/P0.

Both file styles are read: the SasView format (`FileFormatVersion 1.0`, `BEGIN_DATA`, columns `SpinEchoLength
Depolarisation Depolarisation_error … Wavelength … Polarisation Polarisation_error`) and the older Delft files
(`Thickness [cm]: 0.2`, a line of column titles with units in brackets). A missing quantity is derived:
G(δ) − G(0) = ln(P/P0)/(λ²t) from the polarisation, or P/P0 from the depolarisation. The lengths are converted to the
length unit of the plot. An acceptance in the header sets the SESANS resolution (next section).

A SESANS data set keeps its transform: you can switch it between G(δ) − G(0) and P/P0 (the data points follow), but
not to I(Q) or MSAS. The wavelength of every point is used for P/P0 when **λ from file** is ticked; time-of-flight
SESANS, where λ grows with δ, needs it.

The background row is an *offset* of G(δ) − G(0) for SESANS data (ideally zero). The power-law term of the background
is not used.

SANS and SESANS data sets can be fitted together (several data sets, see {doc}`fitting`): the contributions and
global parameters are shared, every data set has its own transform. The plot shows the data sets on the axis of the
current data set (q or δ); choose the data set in "background above is that of data set" to switch. χ² and the fit use
all data sets. The **fit q range** of the Fit panel is not applied to SESANS data.

## SESANS resolution

The resolution of SESANS is the acceptance of the instrument: only neutrons scattered up to Q_max reach the analyser,
so the Hankel transform is clipped there. As in SASfit (which takes the resolution column as the upper limit), *both*
Q integrals end at Q_max:

    G(δ) − G(0) = 1/(2π) ∫_0^Q_max(δ) I(Q) [J0(Qδ) − 1] Q dQ.

Choose it in the **resolution** row of the Transform box:

- **none** — the integrals run over all Q (with a Porod tail above the grid);
- **Q_max** — a fixed value in the unit of Q (from `z-acceptance [nm^-1]` in a file);
- **θ_max [rad]** — the acceptance angle (from `Theta_zmax` in a file); Q_max = 2π sin(θ_max)/λ is computed for every
  spin-echo length with its own wavelength, so in time-of-flight SESANS the cut moves with δ.

The limits are inserted as nodes of the Q grid, so the cut is exact (no interpolation across it); the Q grid then ends
at the largest Q_max. In Python: `SESANSTransform(…, q_accept=0.03)`, `q_accept=<array>` with `accept_delta=<δ>`,
or `theta_max=0.0168` (with `wavelength`/`wavelength_delta`). Tested against direct quadrature of the clipped integral
(`tests/studio/test_sesans_msas.py`): 10⁻⁴ of the largest value.

## Multiple small-angle scattering

Choose **MSAS**, enter the thickness and the wavelength, and say how the data were normalised:

- **incident beam: × exp(−Σt) (SASfit)** — the data were divided by a transmission that does not contain the loss
  by small-angle scattering. This is SASfit's MSAS.
- **measured transmission** — the measured transmission already contains exp(−Σt).

The status line and the Transform box show Σt and the fractions of the beam scattered once and more than once.
For a thin sample the MSAS curve equals I(Q). A data set added while MSAS is shown takes over the MSAS settings.

With MSAS the model is no longer a sum of its contributions: variable projection leaves the scales, the amplitudes
N and the background of such data sets to the optimiser. For SESANS G(δ) − G(0) the model stays linear, and variable
projection works as for I(Q).

## How it is computed

`pysasfit.studio.hankel` is the Python counterpart of SASfit's `Gztransform` and `imMSAStransform` (`sasfit.c`).

**SESANS.** The model is evaluated on a logarithmic Q grid from 0.01/δ_max to 60/δ_min (64 points per decade;
**Q grid / decade** changes it). Between two grid points I(Q)·Q is taken as linear, and every interval is integrated
*exactly* against J0(Qδ) − 1 (with ∫J0 = `scipy.special.itj0y0` and ∫Q J0 = Q J1/δ). The oscillation of J0 therefore
needs no sampling; only I(Q) itself must be resolved by the grid. Below the grid I(Q) is taken as constant, above it
as a Porod tail Q⁻⁴, so that G(0) is complete. The weights depend only on δ and the grid: they are computed once, and
every evaluation in a fit is one matrix product.

**MSAS.** With φ(r) = λ²t G(r), τ = φ(0) = Σt and k0 = 2π/λ (Schelten & Schmatz 1980):

    I_MSAS(Q) = c · k0²/(2π t) ∫ [exp(φ(r)) − 1] J0(Qr) r dr,   c = exp(−τ) (incident beam) or 1 (transmission).

The single-scattering term φ is split off, exp(φ) − 1 = φ + ψ, and evaluated directly as I(Q); only ψ = exp(φ) − 1 − φ
goes through the two transforms. I(Q) is evaluated on a Q grid (Q_min/100 … 30 Q_max, 48 per decade), G(r) on a fine
r grid (384 per decade: only exp(φ) is computed there, but the back transform needs it for a large dynamic range of
I(Q)). Σt > 100 is refused as unphysical (check thickness, wavelength and absolute scale).

**Accuracy** (tests in `tests/studio/test_sesans_msas.py`): the SESANS transform of a sphere of 1 µm agrees with the
analytic projection (Andersson et al. 2008) to 4·10⁻⁴ of G(0); a Gaussian I(Q) agrees with its closed form. The MSAS
of a Gaussian agrees with its series Σ τⁿ/n! … to 0.3 % where the intensity is above 1 % of its maximum, also for
Σt = 0.16; for polydisperse spheres the error is about 10⁻³.

## From Python

```python
from pysasfit.core.sesans_io import load_sesans
from pysasfit.studio import ModelFunction, ScatteringContribution
from pysasfit.studio.fitting import FitProblem
from pysasfit.studio.hankel import make_transform

ses = load_sesans("se008731_01.ses", length_unit="A")         # δ in Å, G in Å⁻²cm⁻¹, t in cm
c1 = ScatteringContribution("ff_sphere", {"R": 9000, "eta": 1})
model = ModelFunction([c1], scales=[2e-18],
                      transform=make_transform("sesans"))      # or "sesans_p", thickness=ses.thickness,
                                                               #    wavelength=ses.wavelength, wavelength_delta=ses.delta
problem = FitProblem()
problem.add_dataset((ses.delta, *ses.observable("G")), model)
problem.add_parameter("0.ff.R", lower=1000, upper=30000)
problem.add_parameter("scale.0", lower=0)
print(problem.fit("scipy", vp=True).text())

# multiple scattering: the thickness can be a fit parameter
msas = ModelFunction([c1], scales=[1e-6], background=0.01,
                     transform=make_transform("msas", thickness=0.2, wavelength=0.6, norm="incident"))
```

Fit addresses of a transform: `transform.thickness`, `transform.wavelength` (a single wavelength). The Copy Python
button writes the transform and the `load_sesans` call of every SESANS data set.

## Not yet

- a fit box for the thickness in the explorer (the engine can fit `transform.thickness`);
- a wavelength distribution for MSAS, and the resolution (dq) together with MSAS in the explorer;
- the Ogata / QDHT / FFTLog strategies of SASfit (`MSAS_QDHT`, `MSAS_FFTLOG`) as alternatives.
