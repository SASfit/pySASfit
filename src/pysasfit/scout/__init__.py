"""pysasfit.scout -- pySASscout, a model search for small-angle scattering data.

Given a measured curve, the scout proposes combinations of form factor, size distribution and (if the data
suggest one) structure factor from the SASfit catalogue, fits each with the Studio's fitting engine and ranks
them with criteria that penalise parameters:

    from pysasfit import scout
    result = scout.search("sample.dat")          # or (q, I, sigma)
    print(result.text())                         # features of the data, ranking, parameters, warnings
    result.save("sample_scout.json")             # the protocol of the run (inputs, seed, versions, all candidates)
    model = result.model(0)                      # studio.ModelFunction of the best candidate

Steps (this first version covers the core search):

1. :mod:`~pysasfit.scout.features` -- Guinier radius, power laws, minima, interference peak, data check;
2. :mod:`~pysasfit.scout.candidates` -- the candidate space, start values and bounds from the features;
3. :mod:`~pysasfit.scout.search` -- multistart fits with variable projection, greedy mixtures;
4. :mod:`~pysasfit.scout.scoring` -- AICc, BIC, Akaike weights, runs test of the residuals;
5. the ranking with warnings and the JSON protocol;
6. :mod:`~pysasfit.scout.store` -- the results store: with ``search(..., store=folder)`` every fit and search is
   kept on disk and reused, so a search can be repeated, continued or extended without fitting again.

The result is a ranking with uncertainty, not a statement about the true model: several candidates often fit
equally well, and the Akaike weights and the notes say so. Command line: ``pysasfit-scout DATA [options]``;
window (also the "pySASscout" tab of the launcher): ``pysasfit-scout-gui`` (:mod:`pysasfit.scout.gui`).
"""

from . import candidates, features, reference, scoring, store
from .candidates import (DISTRIBUTIONS, FORMS, STRUCTURE_FACTORS, Component, DistSpec, FormSpec, Par, SqSpec,
                         register_distribution, register_form, register_structure_factor)  # fmt: skip
from .features import Features, extract
from .search import Candidate, ScoutOptions, ScoutResult, build, fit_candidate, search
from .store import ResultStore

__all__ = [
    "search", "ScoutOptions", "ScoutResult", "Candidate", "fit_candidate", "build",
    "extract", "Features",
    "FORMS", "DISTRIBUTIONS", "STRUCTURE_FACTORS", "Component", "FormSpec", "DistSpec", "SqSpec", "Par",
    "register_form", "register_distribution", "register_structure_factor",
    "ResultStore",
    "candidates", "features", "scoring", "reference", "store",
]  # fmt: skip
