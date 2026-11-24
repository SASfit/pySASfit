# Polarised neutrons

SASfit has one function per polarisation channel of a magnetic model (`_pp`, `_mm`, `_pm`, `_POLARIS`, `_cross`,
`_aniso`, …) and repeats the instrument values (polarisation, flipper efficiency) in every model. In pySASstudio
the **model** describes the sample only and the **data set** says what was measured. One magnetic model then fits
the unpolarised data, flipper off and flipper on, the corrected I+ and I−, their difference and – with an
analyser – the four channels, all at the same time.

## Basis terms

A magnetic model returns, for every q and angle ψ between q and the field on the detector, five terms
(field perpendicular to the beam; N nuclear, M⊥ magnetic Halpern–Johnson amplitude, h the field axis):

| term | meaning |
|---|---|
| NN | ⟨\|N\|²⟩, nuclear |
| NM | ⟨Re N M⊥h*⟩, nuclear–magnetic interference |
| MH | ⟨\|M⊥h\|²⟩, magnetic, non-spin-flip |
| MSF | ⟨\|M⊥\|² − \|M⊥h\|²⟩, magnetic, spin-flip |
| CH | chiral term (0 unless the model is chiral) |

The measured cross sections are weighted sums. **+ and − are the neutron spin parallel / antiparallel to the
field** (convention of Moon, Riste & Koehler, Phys. Rev. 181, 920 (1969): σ_z|±⟩ = ±|±⟩ with z along the field).
The neutron moment is opposite to its spin, so the amplitudes are U±± = b ± p M⊥z: a neutron with spin parallel to
the magnetisation sees nuclear and magnetic amplitude added (this is why a magnetised Co crystal reflects the
neutrons with spin parallel to its moment and to the field). Hence

    I++ = NN + MH + 2NM      I−− = NN + MH − 2NM      I+− = MSF − 2CH      I−+ = MSF + 2CH
    I+  = I++ + I+−          I−  = I−− + I−+          (SANSPOL, no analyser)
    I− − I+ = −4 NM + 4 CH

```{note}
Michels (book and reviews), the SASfit C functions (`_pp`, `_mm`, `pol` = +1) and part of the magnetic-SANS
literature label the states by the neutron **moment**: their "++" (with −2NM) is "−−" here. The `pol` parameter
of the SASfit functions keeps its SASfit meaning; pol = +1 is spin antiparallel to the field. When a data set
has a polarisation setting, C functions with a `pol` parameter get pol = n− − n+ of that setting.
```

For particles whose moments have the orientation moments S1 = ⟨cos θ⟩ and S2 = ⟨P₂(cos θ)⟩ with respect to the
field (nuclear amplitude F_N, magnetic amplitude F_M, same normalisation as I(0) = (Δη V)²):

    NN  = F_N²
    NM  = F_N F_M S1 sin²ψ
    MH  = F_M² [S2 sin⁴ψ + (1 − S2)/3 sin²ψ]
    MSF = F_M² [2(1 − S2)/3 − S2 sin⁴ψ + (4 S2 − 1)/3 sin²ψ]

Saturated: S1 = S2 = 1 (MH = F_M² sin⁴ψ, MSF = F_M² sin²ψ cos²ψ); random: S1 = S2 = 0 (MH + MSF = 2F_M²/3);
Langevin: S1 = L(ξ), S2 = 1 − 3L(ξ)/ξ.

### Magnetic models (Python ports)

| function | model |
|---|---|
| `ff_py_magnetic_shell_psi` | core-shell sphere, nuclear and magnetic SLDs of core, shell and matrix, saturated |
| `ff_py_superparamagnetic_ff_psi` | core-shell sphere with a magnetic core, Langevin statistics with ξ = mu_kT (R/R_av)³ |

Both reproduce the libsasfit originals (`ff_magnetic_shell_psi`, `ff_superparamagnetic_ff_psi`) exactly when no
polarisation is set: then their `pol` parameter mixes the states as in SASfit, I = (1+pol)/2 I− + (1−pol)/2 I+ in the
spin labels used here (pol = +1: spin antiparallel to the field). ψ is the
angle between q and the field (90°: q perpendicular to the field). With a size distribution over R the
Langevin parameter follows the size of each particle.

## What a data set measures

The **Polarisation** box under the Transform box (per data set) sets the quantity:

| quantity | data | instrument values |
|---|---|---|
| not polarised | ordinary data; magnetic functions use their `pol` parameter | – |
| unpolarised | polariser out, or (I+ + I−)/2 | – |
| flipper off | polariser in, flipper off, **not corrected** | P (, P_f) |
| flipper on | polariser in, flipper on, **not corrected** | P, f (, P_f) |
| I+, I− | SANSPOL corrected (pySASred output `_plus`, `_minus`) | – |
| I− − I+ | difference, corrected | – |
| I++, I−−, I+−, I−+ | four channels with analyser, corrected | – |

Only data that are **not corrected** for the instrument need P (beam polarisation), f (flipped fraction of the
flipper, F = 2f − 1) and P_f (polarisation after the sample, depolarisation): the model mixes I+ and I−
exactly as {doc}`pySASred <../sasred/polarisation>` does,

    n+_off = (1 + P r)/2,   n−_on = f n+_off + (1 − f) n−_off,   r = (1 − P_f)/(−ln P_f),

so fitting the uncorrected states with the forward model is equivalent to correcting them first – without the
noise amplification of the inversion. The box shows only the values the quantity needs and where each one comes
from:

- **from the file**: a reduced NXcanSAS file of pySASred carries the polarisation of a SANSPOL output (`_plus`,
  `_minus`, `_unpolarised`, `_difference`) or the polariser/flipper state of the run and P, f from the metadata or
  the SANS-I defaults (Aswal et al. 2008 table, f = 0.995). Adding such a file sets the box.
- **entered**: typed in the box.

Without an analyser P and the magnetic contrast are degenerate (only P·(I+ − I−) is measured): give P from the
instrument; fit it only if the magnetic SLD is known.

Further settings: *flipper off = +* (which flipper state is spin parallel to the field), *exact / BerSANS* flipper
mixing, the **field direction ψ_H** on the detector (from +q_x, counter-clockwise; the ψ of the magnetic functions is
measured from it, so the 2D map shows the field along ψ_H) and **azimuthal average** for 1D radial averages
(sin²ψ → 1/2, sin⁴ψ → 3/8; the ψ parameter is then not used).

Non-magnetic contributions count with the nuclear weight: fully in I+, I−, the non-spin-flip channels and
unpolarised data, **not** in the spin-flip channels and in I− − I+. The background stays a parameter of each data
set (it is not mixed).

C functions that only know SASfit's `pol` parameter (for example the libsasfit originals) get
pol = n+ − n− of the data set, which covers unpolarised, flipper off/on, I+ and I−; for the difference and the
analyser channels a Python port with basis terms is needed.

## Fitting several states together

Load the states as data sets (flipper off and on, or I+ and I−, optionally the polariser-out run), set the
quantity of each and fit the magnetic parameters as common parameters. P, f, P_f and ψ_H can be fitted from a
script with the addresses `pol.P`, `pol.f`, `pol.P_f`, `pol.psi_H`:

```python
import numpy as np
from pysasfit.studio import ModelFunction, ScatteringContribution, ports
from pysasfit.studio.fitting import FitProblem
from pysasfit.studio.polarisation import Polarisation

ports.register_aliases()
c = ScatteringContribution("ff_py_superparamagnetic_ff_psi",
                           {"R": 5, "dR": 1, "eta_nuc_core": 6.9, "eta_nuc_shell": 0.5, "eta_nuc_matrix": 6.0,
                            "eta_mag_core": 1.5, "R_av": 5, "mu_kT": 5, "psi": 90},
                           size_distribution="sd_LogNorm", sd_params={"N": 1, "s": 0.2, "p": 1, "mu": 5},
                           distributed="R")
m_off = ModelFunction([c], background=0.01, polarisation=Polarisation("off", P=0.984, f=0.995))
m_on = ModelFunction([c], background=0.01, polarisation=Polarisation("on", P=0.984, f=0.995))

prob = FitProblem()
prob.add_dataset("sample_off.h5", m_off, name="off")
prob.add_dataset("sample_on.h5", m_on, name="on")
prob.add_common("0.ff.eta_mag_core")
prob.add_common("0.sd.mu")
prob.add_local("background")
print(prob.fit("scipy").text())
```

A model may also be given the setting as a dictionary (`polarisation={"quantity": "minus"}`); this is what
*Copy Python code* writes.

## Conventions

The pySASfit frame ({doc}`../conventions`): q_x to the right and q_y up as seen looking towards the sample,
ψ counter-clockwise from +q_x. With the field along +q_x (ψ_H = 0) the frame of Michels (field along z_M) is
x_M = z, z_M = x, y_M = −y, and the geometric factors are identical. + and − refer to the neutron spin (Moon, Riste
& Koehler 1969); Michels and SASfit label by the neutron moment (their "++" is the "−−" here).
