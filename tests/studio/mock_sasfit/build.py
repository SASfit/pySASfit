"""Build the mock SASfit libraries used by the studio tests (needs a C compiler)."""

import os
import shutil
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def build(dest):
    """Compile libsasfit + two plugins into ``dest`` (layout of a SASfit installation)."""
    ext = "dll" if sys.platform == "win32" else ("dylib" if sys.platform == "darwin" else "so")
    cc = shutil.which("gcc") or shutil.which("cc")
    if cc is None:
        return None
    os.makedirs(os.path.join(dest, "plugins"), exist_ok=True)
    jobs = [("libsasfit.c", os.path.join(dest, f"libsasfit.{ext}"))]
    for name in ("mockplug", "mockdep"):
        jobs.append((f"libsasfit_{name}.c", os.path.join(dest, "plugins", f"libsasfit_{name}.{ext}")))
        shutil.copy(os.path.join(HERE, f"sasfit_{name}.h"), os.path.join(dest, "plugins"))
    for src, out in jobs:
        r = subprocess.run([cc, "-shared", "-fPIC", "-O2", "-o", out, os.path.join(HERE, src), "-lm"],
                           capture_output=True, text=True)
        if r.returncode:
            raise RuntimeError(r.stderr)
    return dest


if __name__ == "__main__":
    print(build(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "build")))
