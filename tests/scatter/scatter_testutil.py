"""Helpers shared by the pySCAtter tests (Schulz-Zimm weights for brute-force averages)."""
import numpy as np


def gam_w(sigma, s, n=1500):
    z = (1 - sigma ** 2) / sigma ** 2
    r = np.linspace(max(1e-3, 1 - 7 * sigma), 1 + 7 * sigma, n)
    w = np.exp((z + s) * np.log(r) - (z + 1) * r)
    return r, w / np.trapezoid(w, r)
