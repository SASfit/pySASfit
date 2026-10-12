"""pysasfit.studio -- the SASfit plugin functions in Python (from SASstudio).

Gives Python access to all model functions of the compiled SASfit plugins
(form factors, structure factors, size distributions, peaks), described by
the Doxygen comments of the plugin headers, plus Python plugins, and
combines them into models that can be fitted.

    from pysasfit import studio
    studio.search("sphere", "shell")            # catalogue from the headers (any platform)
    print(studio.info("ff_sphere"))
    sphere = studio.function("ff_sphere")        # needs the SASfit libraries (pysasfit-studio install-binaries)
    I = sphere(q, R=50.0, eta=1.0)

Origin: SASstudio (github.com/ISISNeutronMuon/SASstudio), copied into pySASfit
and reworked: plugins are initialised with do_init() and the libsasfit
common-function table, the sasfit_param layout matches sasfit_function.h,
libsasfit precision settings get the SASfit defaults, paths no longer depend
on the working directory, and the size-distribution integration and fitting
are implemented.
"""

from . import headers, paths
from .functions import SASfitError, SASfitFunction, function, register_python_function, unregister_python_function
from .headers import PLUGIN_TYPES, ParameterInfo, PluginInfo, catalogue, info, search
from .library import DEFAULT_PRECISION, SASfitLibrary, SASfitLibraryNotFound, get_library
from .model import FitResult, ModelFunction, Pipeline, ScatteringContribution

__all__ = [
    "headers",
    "paths",
    "catalogue",
    "search",
    "info",
    "PluginInfo",
    "ParameterInfo",
    "PLUGIN_TYPES",
    "function",
    "SASfitFunction",
    "SASfitError",
    "register_python_function",
    "unregister_python_function",
    "SASfitLibrary",
    "SASfitLibraryNotFound",
    "get_library",
    "DEFAULT_PRECISION",
    "ScatteringContribution",
    "ModelFunction",
    "Pipeline",
    "FitResult",
    "available",
]


def available():
    """True if the compiled SASfit libraries are installed for this platform."""
    return paths.lib_dir() is not None
