# Building Robertus SHS and RY Polydisperse Yukawa Core Libraries

## Status

The GUI tabs for **Robertus SHS** and **RY Polydisperse Yukawa** have been added to oZgui.py, but they require compiled C libraries to function. The current SASFIT plugin DLLs do not include the low-level solver functions needed by the Python wrappers.

## What's Needed

To make these tabs active, you need to build the core C libraries from the plugin source code:

### Windows Build (MinGW)

```bash
cd src/plugins/robertus_shs

# Build the core library (not the plugin)
gcc -O2 -fPIC -shared -o ../../pyozGUI/librobertus.dll \
    robertus_shs_core.c \
    -lgsl -lgslcblas -lsundials_kinsol -lsundials_nvecserial -lm
```

```bash
cd src/plugins/ry_polydisperse_yukawa

# Build the RY core library
gcc -O2 -fPIC -shared -o ../../pyozGUI/libry_polydisperse.dll \
    ry_polydisperse_core.c \
    -lgsl -lgslcblas -lsundials_kinsol -lsundials_nvecserial -lm
```

### Linux Build

```bash
cd src/plugins/robertus_shs

gcc -O2 -fPIC -shared -o ../../pyozGUI/librobertus.so \
    robertus_shs_core.c \
    -lgsl -lgslcblas -lsundials_kinsol -lsundials_nvecserial -lm
```

```bash
cd src/plugins/ry_polydisperse_yukawa

gcc -O2 -fPIC -shared -o ../../pyozGUI/libry_polydisperse.so \
    ry_polydisperse_core.c \
    -lgsl -lgslcblas -lsundials_kinsol -lsundials_nvecserial -lm
```

## Alternative: Set Environment Variables

Instead of placing the libraries in pyozGUI/, you can build them anywhere and set:

- `ROBERTUS_SHS_LIB=/path/to/librobertus.dll`
- `RY_POLYDISPERSE_LIB=/path/to/libry_polydisperse.dll`

## Dependencies

Both libraries require:
- **GSL** (GNU Scientific Library) - libgsl and libgslcblas
- **SUNDIALS 7.x** - libsundials_kinsol and libsundials_nvecserial
  - These are available in `src/sundials7/`
  - The main SASfit build should have already built these

## Tab Features

Once built, the new tabs provide:

### Robertus SHS Tab
- Size distributions: Log-normal, Schulz-Zimm, Gamma, Beta
- Form factors: Sphere or Core-shell
- Interaction: Sticky hard sphere (Baxter stickiness parameter tau)
- Exact partial structure factors S_ij(Q)
- All six SASfit approximation schemes compared to exact I(Q)
- Plot I(Q), approximation errors, S_ij(Q), size distribution
- Export to CSV and Excel

### RY Polydisperse Yukawa Tab
- Size distributions: Log-normal, Schulz-Zimm, Gamma, Beta
- Interaction: Rogers-Young Yukawa (screening z, contact parameter K, alpha)
- Exact partial structure factors S_ij(Q)
- All six SASfit approximation schemes
- Plot S(Q), S_ij(Q), size distribution
- Export to CSV and Excel

## Testing

Once built, launch the GUI:
```bash
python src/pyozGUI/oZgui.py
```

The new tabs should appear in the notebook next to "Polydisperse Yukawa" and "OZ Solver".

If the libraries aren't found, the tabs will show an error message with detailed build instructions.

## Troubleshooting

### "function 'rshs_make_classes' not found"
This means the library was found but is the wrong one (likely the SASFIT plugin DLL). Make sure you're building from `*_core.c`, not the plugin build system.

### "Could not find module ... or one of its dependencies"  
The library requires GSL and SUNDIALS. Make sure these dependencies are available:
- On Windows, ensure libgsl-28.dll, libgslcblas-0.dll are in PATH or same directory
- On Linux, install GSL and SUNDIALS development packages

### Library not found at all
- Check that the built library is in src/pyozGUI/ or one of the alternative locations
- Check that the filename matches what the wrapper expects (librobertus.dll or librobertus.so)
- Set the environment variable explicitly as a fallback

## Development Notes

- **robertusWrapper.py**: ctypes wrapper around the C core library
- **robertus_shs_sas.py**: Python SAS layer using the wrapper
- **robertus_shs_tab.py**: GUI tab implementation
- **rypolydisperseWrapper.py**: ctypes wrapper for RY solver
- **ry_polydisperse_yukawa_sas.py**: Python SAS layer for RY (partial - needs completion)
- **ry_polydisperse_yukawa_tab.py**: GUI tab implementation

The wrappers expect the low-level C API (rshs_*, ry_polydisperse_*) functions, not the high-level SASFIT plugin interface.
