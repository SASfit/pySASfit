"""
Fit a triangular-shaped function to 1D data.

Usage examples (in a notebook where `SUMtof12000` exists):

import numpy as np
from fit_triangle import fit_triangle, triangle_model
x = np.arange(len(SUMtof12000))
params, cov = fit_triangle(x, SUMtof12000)

# params = (center, width, amplitude, baseline)

Or run the script directly to demo with synthetic data:
python fit_triangle.py

Dependencies: numpy, scipy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit


def triangle_model(x, center, width, amplitude, baseline):
    """Triangular pulse model.

    Parameters
    - x: array-like
    - center: center position of the triangle (same units as x)
    - width: full base width of the triangle (w>0)
    - amplitude: peak height above baseline
    - baseline: constant offset

    Triangle shape (value at center = amplitude + baseline; linearly falls to baseline at +/-width/2):
    value(x) = baseline + amplitude * max(0, 1 - 2*abs(x-center)/width)
    """
    x = np.asarray(x, dtype=float)
    w = float(width)
    if w <= 0:
        # Avoid division by zero; return baseline
        return np.full_like(x, baseline)
    tri = np.maximum(0.0, 1.0 - 2.0 * np.abs(x - center) / w)
    return baseline + amplitude * tri


def fit_triangle(x, y, p0=None, bounds=None, plot=True, ax=None):
    """Fit a triangular model to y(x).

    Parameters
    - x, y: array-like of same length
    - p0: optional initial guess (center, width, amplitude, baseline)
    - bounds: optional bounds for parameters as (lower, upper) for curve_fit
    - plot: if True, create a plot of data + fit
    - ax: optional matplotlib Axes to plot into

    Returns
    - popt: best-fit parameters (center, width, amplitude, baseline)
    - pcov: covariance matrix from curve_fit
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    y=y/np.max(y)
    if x.shape != y.shape:
        raise ValueError("x and y must have the same shape")

    # sensible defaults for initial guess
    if p0 is None:
        # center ~ weighted mean or location of maximum
        center0 = x[np.argmax(y)]
        # width ~ half-range or FWHM-ish: take where y falls below half of peak
        peak = np.max(y)
        baseline0 = np.percentile(y, 5)
        amp0 = peak - baseline0
        # approximate width: find indices where y > baseline + amp0/2
        half_mask = y > (baseline0 + amp0 / 2.0)
        if np.any(half_mask):
            xs = x[half_mask]
            width0 = (xs.max() - xs.min()) * 1.5 if xs.size > 1 else max(1.0, (x.max() - x.min()) / 10.0)
        else:
            width0 = max(1.0, (x.max() - x.min()) / 10.0)
        p0 = (center0, width0, amp0 if amp0 > 0 else (y.max() - y.min()), baseline0)

    # sensible bounds if not provided
    if bounds is None:
        lower = [x.min(), 1e-8, -np.inf, -np.inf]
        upper = [x.max(), x.max() - x.min() + 1e-8, np.inf, np.inf]
        bounds = (lower, upper)

    # perform fit
    try:
        popt, pcov = curve_fit(triangle_model, x, y, p0=p0, bounds=bounds)
    except Exception as e:
        raise RuntimeError(f"triangle fit failed: {e}")

    if plot:
        # create plot
        if ax is None:
            fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        ax.plot(x, y, marker='.', linestyle='none', alpha=0.6)
        xf = np.linspace(x.min(), x.max(), max(400, x.size * 4))
        yf = triangle_model(xf, *popt)
        ax.plot(xf, yf,  color='C1')
        ax.axvline(popt[0], color='C1', ls='--', alpha=0.6)
        ax.set_xlabel('tof [ms]')
        ax.set_ylabel('Intensity (normalized)')
        ax.legend()
        plt.tight_layout()

    return popt, pcov


if __name__ == '__main__':
    # Demo with synthetic data
    import numpy as np

    rng = np.random.default_rng(12345)
    x = np.linspace(0, 100, 201)
    # true params
    true = (40.0, 30.0, 10.0, 2.0)  # center, width, amp, baseline
    y_true = triangle_model(x, *true)
    # add noise
    y = y_true + rng.normal(scale=0.8, size=x.shape)

    popt, pcov = fit_triangle(x, y, plot=True)
    print('fitted params (center,width,amp,baseline)=', popt)
    plt.show()
