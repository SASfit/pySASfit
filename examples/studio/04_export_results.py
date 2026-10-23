"""Fit, then save the result as Excel workbook, CSV files and summary figure, with the size distributions.

    python examples/studio/04_export_results.py          # writes into examples/studio/output/
"""

import os
import runpy

from pysasfit.studio import export

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "output")
os.makedirs(OUT, exist_ok=True)

# the problem of example 03 (without its plot): run it as a module and take its objects
ns = runpy.run_path(os.path.join(HERE, "03_fit_with_engine.py"), run_name="example03")
prob, res = ns["prob"], ns["res"]

# --8<-- [start:export]
p = 3.0                                              # weight: N(x) * x**p is exported next to N(x); 3 = volume weight
export.save_xlsx(os.path.join(OUT, "fit.xlsx"), res, prob, dist_power=p)    # needs openpyxl
export.save_csv(os.path.join(OUT, "fit.csv"), res, prob, dist_power=p)      # fit_parameters.csv, fit_curve.csv, fit_distributions.csv, ...
export.save_figure(os.path.join(OUT, "fit.pdf"), res, prob, dist_power=p)   # .png and .svg work as well
png = export.figure_bytes(res, prob, fmt="png", dist_power=p)               # bytes, e.g. for a report
# --8<-- [end:export]

print("written to", OUT, f"({len(png) / 1024:.0f} kB picture)")
