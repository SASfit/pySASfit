"""
Helper utilities for fitting stacks of TOF sums.

Provides `fit_stack` which fits a triangular peak model to a stack of
SUMtof arrays. The function is written to be importable from notebooks
and scripts and avoids use of `globals()` by accepting `tof` explicitly.
"""
import numpy as np
from scipy import constants as _scipy_constants
import matplotlib.pyplot as plt
from pySASfit.io_tools.SASformats import SANSdata

def read_files_sumtof(file_tofs):
    sum_tofs=[]
    for fn in file_tofs:
        #print(fn)
        SASD=SANSdata(fn)
        try:
#            print(SASD.BerSANS["%File,DataSizeTOF"], SASD.BerSANS["%Setup,LambdaC"],SASD.BerSANS["%Setup,Collimation"])
            DetData = SASD.BerSANS["%Counts,DetCounts"]
            sum_tofsi=[]
            for i in range(SASD.BerSANS["%File,DataSizeTOF"]):
                #print(np.sum(DetData[:,:,i]))
                sum_tofsi.append(np.sum(DetData[:,:,i]))
            sum_tofs.append(sum_tofsi)
        except Exception:
            pass
    return sum_tofs
        
def fit_stack(sum_tofs, labels, tof=None, L=None, rpm_start=12000, rpm_step=2000,
              plot=False, other_centers=None, constants_module=None):
    """Fit a stack of SUMtof arrays using the triangular model.

    Parameters
    - sum_tofs: iterable of 1D arrays (n_files x ntof)
    - labels: iterable of names (same length as sum_tofs)
    - tof: optional x-axis array matching each y's length; if None, index axis used
    - L: optional flight path length used to compute lambda; if None, lambda may be
         calculated from `other_centers` if provided
    - rpm_start, rpm_step: used to fill `rpm` field in results
    - plot: if True, calls the underlying `fit_triangle` with plot=True
    - other_centers: optional list of center values used to compute lambda for
                     paired datasets (preserves previous notebook behavior)
    - constants_module: optional module providing `Planck` and `neutron_mass` etc.

    Returns list of dicts with fit results (same schema as before).
    """
    try:
        # Local import to avoid heavy imports at module import time
        from pySASfit.io_tools.fit_triangle import fit_triangle, triangle_model
    except Exception:
        raise ImportError("fit_triangle is required by fit_helpers.fit_stack")

    consts = constants_module if constants_module is not None else _scipy_constants

    results = []
    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    for idx, (y_raw, name) in enumerate(zip(sum_tofs, labels)):
        y = np.asarray(y_raw, dtype=float)

        # Determine x-axis
        if tof is not None and len(tof) == y.size:
            x = np.asarray(tof, dtype=float)
        else:
            x = np.arange(y.size, dtype=float)

        # Pre-clean NaNs
        nan_mask = np.isnan(y)
        if np.all(nan_mask):
            print(f"{name}: all NaN, skipping")
            continue
        if np.any(nan_mask):
            x_fit = x[~nan_mask]; y_fit = y[~nan_mask]
        else:
            x_fit = x; y_fit = y

        try:
            popt, pcov = fit_triangle(x_fit, y_fit, plot=plot,ax=ax)
            perr = np.sqrt(np.diag(pcov)) if pcov is not None else [np.nan]*4
            center, width, amplitude, baseline = popt
            center_err, width_err, amp_err, base_err = perr

            # Dense model evaluation (returned only if needed later)
            xf = np.linspace(x.min(), x.max(), max(400, x.size*4))
            yf = triangle_model(xf, *popt)

            # Compute lambda using either L or other_centers (backwards compatibility)
            if other_centers is not None and idx < len(other_centers) and L is None:
                lamb = (other_centers[idx] - center) / 4000.0 * consts.Planck / consts.neutron_mass / consts.nano
                SD=center / lamb * consts.Planck / consts.neutron_mass / consts.nano
            elif L is not None:
                lamb = center / L * consts.Planck / consts.neutron_mass / consts.nano
                SD=L
            else:
                lamb = None

            results.append({
                "name": name,
                "center": center, #"center_err": center_err,
                "width": width, #"width_err": width_err,
                #"amplitude": amplitude, "amplitude_err": amp_err,
                #"baseline": baseline, "baseline_err": base_err,
                "rpm": (rpm_start + idx * rpm_step),
                "lambda": lamb, "lambda_err":lamb*center_err/center,
                "DL/L": width / center / 2.0 if center != 0 else np.nan,
                "DL/L_err": width_err / center / 2.0 if center != 0 else np.nan,
                "SD":SD,
            })
            if plot:
                ax.plot(xf, yf, label=name)

        except Exception as exc:
            print(f"{name}: fit failed: {exc}")
            results.append({
                "name": name,
                "center": np.nan, "center_err": np.nan,
                "width": np.nan, "width_err": np.nan,
                "amplitude": np.nan, "amplitude_err": np.nan,
                "baseline": np.nan, "baseline_err": np.nan,
                "rpm": (rpm_start + idx * rpm_step),
                "lambda": None,
                "DL/L": np.nan,
            })

    return results
