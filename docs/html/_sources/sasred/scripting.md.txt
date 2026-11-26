# Scripts and output files

## Reducing a run from Python

```python
from pysasfit.sasred.io.sinq import read_sans_i
from pysasfit.sasred.core.beamcentre import masked_centre_of_gravity
from pysasfit.sasred.core.averaging import q_edges
from pysasfit.sasred.core import mask as mk
from pysasfit.sasred.reduce import reduce_run_1d
from pysasfit.sasred.io.nxcansas import write_nxcansas_1d
from pysasfit.sasred.io.ascii import write_ascii_1d

raw_dir = "/data/sans-i/2026/Vifor"
bc = masked_centre_of_gravity(read_sans_i(f"{raw_dir}/sans2026n082173.hdf").counts)   # direct beam
centre = (bc.x, bc.y)                                         # pixels, from 0

raw = read_sans_i(f"{raw_dir}/sans2026n082130.hdf")
mask, report = mk.suggest_mask(raw.counts, centre)            # border + beamstop + hot pixels
edges = q_edges(0.006, 0.09, 60)                              # internal unit: 1/Å (0.06 … 0.9 1/nm)
r = reduce_run_1d(raw, centre, edges, mask=mask)

print(r.q, r.I, r.idev())                                     # q in 1/Å, I in counts/monitor/sr
print({k: v.max() for k, v in r.intensity.budget().items()})  # variance share per source
write_nxcansas_1d("A_1_8m.h5", r)                             # Q in 1/nm (units="A" for 1/Å)
write_ascii_1d("A_1_8m.dat", r)
```

Internally the reduction works in Å and 1/Å (`raw.wavelength_A`, `r.q`, `q_edges`). The writers convert to the
output units, nm and 1/nm by default; {mod}`pysasfit.sasred.units` converts in scripts:

```python
from pysasfit.sasred.units import units
u = units("nm")
u.q_out(r.q)          # 1/Å -> 1/nm
u.q_in(0.5)           # 0.5 1/nm -> 1/Å
u.lam(raw.wavelength_A)
```

`reduce_run_1d` takes further `normalise_by="time"`, `sector_deg=(centre, half_width)` and an `instrument`
dictionary (default: the SANS-I file). The result `ReducedRun` holds, besides q, I and Idev, the 2D corrected
image (`r.image`, a `UArray`), the binning, the pixel geometry, the dead-time factor, the parameters and the
notes.

The run table of the GUI is available as {func}`pysasfit.sasred.catalogue.scan_folder` (headers of a whole folder;
without the checksum, so it is fast).

## NXcanSAS file

`write_nxcansas_1d` writes one `SASentry` with:

| item | content |
|---|---|
| `sasdata01/Q`, `Qdev` | q and its resolution, units `1/nm` or `1/angstrom`. `Qdev` (and `Q@resolutions`) only when the geometry has a direct-beam measurement; without it the curve has no resolution and a note says so |
| `sasdata01/I`, `Idev` | intensity and its **statistical** standard deviation (`I@uncertainties = "Idev"`) |
| `I_covariance_statistical` | n × n covariance of the statistical sources (counts) |
| `I_systematic/<source>` | sensitivity dI/ds of every systematic source, with its σ and group (fully correlated scale terms such as the monitor) |
| `I_budget/<source>` | variance contribution of every source at every q (they add up to the total) |
| `sasprocess01` | program and version, date, raw file and its SHA-256, the parameters (beam centre, normalisation, dead time, attenuator, sector, …), the notes |

Standard readers use I, Idev, Q and Qdev; the extra items are inside `SASdata` and do not disturb them.

## ASCII file

Header lines starting with `#` (program, run, sample, raw file, unit of I, parameters, notes, wavelength), then
five columns:

```text
# q [1/nm]  I  Idev(statistical)  Qdev [1/nm]  sigma_systematic
```

`sigma_systematic` is the standard deviation from the systematic sources (monitor, …), which is common to all
points of the curve. Without a direct-beam measurement of the geometry, `Qdev` is 0 and a header line says
`Qdev = 0: resolution not determined`.

The 2D NXcanSAS file (`write_nxcansas_2d`) carries, when the beam profile is known, `Qxdev` and `Qydev`
(`Qx@resolutions`, `Qy@resolutions`) plus `Qdev_parallel`, `Qdev_perpendicular` and `Qxy_correlation` per pixel
(see {doc}`reduction`, *Resolution*). In Python the profiles come from
{func}`pysasfit.sasred.core.beamprofile.beam_profiles` and are passed as `beam_profile=` to `reduce_row`.
