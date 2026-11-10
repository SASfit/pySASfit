"""Helpers for the "Copy Python code" tests: run a generated script without its plot block."""
import os


def run_script(code, tmp_path=None):
    """exec the script with PLOT = False in a fresh namespace (cwd: tmp_path); returns the namespace."""
    assert "PLOT = True" in code
    code = code.replace("PLOT = True", "PLOT = False", 1)
    compile(code, "<python_code>", "exec")
    ns = {"__name__": "__python_code__"}
    old = os.getcwd()
    if tmp_path is not None:
        os.chdir(tmp_path)
    try:
        exec(code, ns)
    finally:
        os.chdir(old)
    return ns


def run_plot(code, tmp_path=None):
    """exec the whole script, plot block included (Agg backend, plt.show replaced)."""
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    show = plt.show
    plt.show = lambda *a, **k: None
    try:
        ns = {"__name__": "__python_code__"}
        old = os.getcwd()
        if tmp_path is not None:
            os.chdir(tmp_path)
        try:
            exec(code, ns)
        finally:
            os.chdir(old)
        plt.close("all")
        return ns
    finally:
        plt.show = show
