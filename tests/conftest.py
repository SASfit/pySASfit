"""Every test runs with factory-default preferences: the user's own
settings file (Settings -> Preferences) must not change test results."""
import os
import tempfile

_tmp = tempfile.mkdtemp(prefix="pysasfit-settings-")
os.environ["PYSASFIT_SETTINGS"] = os.path.join(_tmp, "settings.json")
