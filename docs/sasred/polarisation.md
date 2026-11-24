# Polarised data without spin analysis (SANSPOL)

SANS-I can polarise the incident beam: the polariser (a supermirror cavity in the first collimation
section) is **in or out**, the RF flipper before the sample is **on or off**. There is no analyser behind
the sample. SANS-LLB has no polarisation option. Every state is measured and reduced as a run of its own
(one HDF file each):

| state | polariser | flipper | measures |
|---|---|---|---|
| `unpolarised` | out | – | I_u = (I+ + I−)/2 |
| `off` | in | off | I_off = n+_off I+ + n−_off I− |
| `on` | in | on | I_on = n+_on I+ + n−_on I− |

I+ and I− are the cross sections for a fully polarised beam with the neutron **spin** parallel (+) or
antiparallel (−) to the field at the sample, summed over the final spin states (I± = I±± + I±∓). This is the
convention of Moon, Riste & Koehler, Phys. Rev. 181, 920 (1969): the amplitudes are U±± = b ± p M⊥z, so a spin
parallel to the magnetisation adds the magnetic amplitude, I++ = NN + MH + 2NM, I−− = NN + MH − 2NM, and
I− − I+ = −4 NM (+ 4 CH in chiral samples) is the nuclear–magnetic interference; (I+ + I−)/2 is the unpolarised
cross section.

```{note}
Michels, the SASfit C functions and part of the magnetic-SANS literature label the states by the neutron
**magnetic moment**, which is antiparallel to the spin: their "++" is the "−−" here (Quan et al. 2019).
pySASred and pySASstudio label by the spin. Which flipper state gives spin + is set in the instrument file
(`flipper_off_is_plus`, default: flipper off = +, to be confirmed for SANS-I) and can be changed per data set;
a sample with a known sign of N·M settles it (I− − I+ < 0 for N·M > 0).
```

## Polariser and flipper state of a run

- **Polariser**: stored in the raw file, `entry1/SANS/polarizer/state` ("in" / "out"; read as `raw.polariser`,
  empty for older files without the entry).
- **Flipper**: *not* stored in the raw files. pySASred takes it
  1. from what was set by hand in the run table: right mouse button → **Flipper state** → flipper off / on, or
     *alternate* (off, on, off, … or on, off, … in run order, for the usual alternating measurement); kept per
     raw-data folder like the corrected sample names (`names.py`, also after *Rescan*);
  2. otherwise from the sample name: "flipper on", "flip_off", "flipper=on" or a trailing "_on" / "-off" /
     " (on)" (a plain "on" inside a name, as in "PNIPAM on silica", is not used).

  Runs with the polariser in and an unknown flipper state are shown in orange in the run table, and the
  reduction plan says so ("flipper off and on runs would otherwise be summed").

The grouping keeps polariser in / out and flipper off / on apart (`polariser` and `flipper` are part of the
default matching), the rows of polarised samples are labelled "name [flipper off]" / "name [flipper on]" (the
transmission and the sample position are matched by the plain name), a row that sums runs of different states
gets a warning, and the reduced files carry `polariser`, `flipper` and `flipper_source` in their metadata, from
where the correction and pySASstudio take the state.

## Efficiencies

| symbol | meaning | relation |
|---|---|---|
| P | beam polarisation | p = (1 + P)/2 neutrons in the + state with flipper off; flipping ratio (1 + P)/(1 − P) |
| f | flipped fraction of the flipper | flipper "efficiency" F = 2f − 1 (Quan et al. quote F = 0.99, i.e. f = 0.995) |
| P_f | polarisation after the sample (depolarisation) | r = (1 − P_f)/(−ln P_f); P_f = 1: no depolarisation |

With Pe = P r:

    n+_off = (1 + Pe)/2,           n−_off = (1 − Pe)/2
    n−_on  = f n+_off + (1 − f) n−_off,   n+_on = 1 − n−_on            (exact)

BerSANS (Keiderling et al. 2008, eq. 3b) uses n−_on = p f, which leaves out the (1 − p)(1 − f) term; the
option `form="bersans"` reproduces data corrected that way.

**Depolarisation** (Quan et al., IUCrJ 8 (2021) 455): in a ferromagnet that is not saturated the beam
polarisation decays exponentially along the path, P(x) = P exp(−D x). Neutrons scattered at depth x see
the local polarisation, so for SANSPOL the incident polarisation is replaced by its depth average P r
(P_f = 0.9 → r = 0.949, 0.75 → 0.869). P_f is measured in transmission, which needs an analyser; without
it P_f = 1.

### Where the values come from

1. The **metadata** of the reduced file (header items of the raw run), with the keys listed in the
   instrument file under `polarisation: metadata_keys` (SANS-I: `polariser` / `entry1/SANS/polarizer/state`,
   `flipper`). P and f are not stored in the SANS-I files.
2. Otherwise the **instrument defaults** (`instruments/sinq/sans_i.yaml`, section `polarisation`):
   - P(λ, collimation) from the table of Aswal et al., NIM A 586 (2008) 86, Fig. 4 (proton spin filter,
     δP/P ≤ 0.5 %), interpolated linearly in λ and in log(collimation). Outside the measured range the
     nearest value is used and the source says "extrapolated". A collimation longer than 15 m is
     flagged: the polariser sits in the first collimation section and cannot be in.

     | λ [nm] | 2 m | 3 m | 6 m | 15 m |
     |---|---|---|---|---|
     | 0.47 | 0.957 | 0.960 | 0.973 | 0.977 |
     | 0.60 | 0.976 | 0.981 | 0.984 | 0.984 |
     | 0.80 | 0.982 | 0.983 | 0.987 | – |
     | 1.00 | 0.985 | 0.984 | – | – |

   - f = 0.995 (F = 0.99).
3. Values **entered** by the user win over both.

Every value carries its source, which is written into the notes and the output files.

```{important}
Without an analyser **P cannot be determined from the data**: only the product P (I+ − I−) is measured.
A wrong P scales I+ − I− by 1/P and leaves (I+ + I−)/2 unchanged. f, on the other hand, follows from the
three states (`flipped_fraction_from_data`), provided the polariser-out data are on the same absolute
scale.
```

## Procedure

1. **Reduce every state as usual**: same mask, binning (1D) or pixel grid (2D), background and empty
   cell measured in the same state, transmission, water or direct beam. The correction is linear and is
   applied to the **fully corrected** data, never to raw counts (Keiderling et al. 2008). Sample
   transmissions of the two flipper states agree unless the sample absorbs spin-dependently.
2. **Combine the states** (two or all three):

       I+ − I− = (I_off − I_on) / (f Pe)
       I_u     = I_off − (I_off − I_on) / (2 f)          (independent of P)
       I±      = I_u ± (I+ − I−)/2

   With all three states I+ and I− follow per point from weighted least squares (weights 1/variance of
   each state); chi² per point (one degree of freedom, median ≈ 0.45 when consistent) checks P, f and the
   scale of the polariser-out data.
3. **Outputs**: I+, I−, (I+ + I−)/2 and I− − I+, each as a reduced data set (2D image or 1D curve) with
   the union of the masks.

**Uncertainties**: every source of the reduced states is kept and propagated exactly to first order;
sources shared by the states (the same background, Cd or water run) are counted once. P, f and P_f enter
as systematic scalar sources (group `polarisation`) when their uncertainty is known (the SANS-I table:
σ_P = 0.005); otherwise "uncertainty unknown, not propagated" is noted. For files the inputs carry only
Idev, so the outputs have the statistical Idev; the covariance and systematic sensitivities of the inputs
are removed from the output files because they belong to the uncorrected intensity.

For **fitting** a model it is better to fit the measured states directly with the forward model
(`sanspol_forward`, pySASstudio polarisation box): no noise amplification by the inversion. The
corrected I+, I− and I− − I+ are for display and model-free analysis.

## Scripting

From reduced files (written by pySASred, 1D or 2D):

```python
from pysasfit.sasred.polarisation import correct_files

out = correct_files({"off": "sample_off.h5", "on": "sample_on.h5",
                     "unpolarised": "sample_unpol.h5"},      # optional third state
                    out_prefix="sample_sanspol")
# -> sample_sanspol_plus.h5, _minus.h5, _unpolarised.h5, _difference.h5
```

P and f come from the metadata, else from the SANS-I defaults; give them explicitly with
`correct_files(..., P=0.975, f=0.99, P_f=0.9)` or a full setup:

```python
from pysasfit.sasred.polarisation import PolarisationSetup, sanspol_correct, correct_reduced

setup = PolarisationSetup(P=0.984, f=0.995, sigma_P=0.005, P_f=1.0,
                          form="exact", flipper_off_is_plus=True)
print(setup.describe())

# reduced results of one session (Reduced2D or ReducedRun): all sources stay correlated
res = correct_reduced(setup, off=r_off, on=r_on, unpolarised=r_unpol)
res["difference"].image          # UArray, I- - I+ per pixel
res["result"].chi2               # with three states

# arrays or UArrays directly
r = sanspol_correct({"off": I_off, "on": I_on}, setup)
r.plus, r.minus, r.unpolarised, r.difference
```

Default P of SANS-I for a setting:

```python
from pysasfit.sasred.polarisation import beam_polarisation_table
P, sigma, source = beam_polarisation_table(wavelength_nm=0.6, collimation_m=6)   # 0.984, 0.005
```

## References

- Keiderling et al., *Meas. Sci. Technol.* **19** (2008) 034009 – SANSPOL/POLARIS correction at V4 (BerSANS).
- Aswal et al., *Nucl. Instrum. Methods A* **586** (2008) 86 – SANS-I polariser, flipper and beam polarisation.
- Quan et al., *J. Phys.: Conf. Ser.* **1316** (2019) 012010 – POLARIS at SANS-I with a proton spin filter;
  notation of the spin states.
- Quan et al., *IUCrJ* **8** (2021) 455 – depolarisation correction, optimum sample thickness.
- R. M. Moon, T. Riste, W. C. Koehler, *Phys. Rev.* **181** (1969) 920 – polarisation analysis; the spin
  convention used here.
- A. Michels, *Magnetic Small-Angle Neutron Scattering*, Oxford University Press (2021).
