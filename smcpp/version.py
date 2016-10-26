from setuptools_scm import get_version

try:
    __version__ = get_version()
except:
    __version__ = "0.0.0"

if "+" in __version__:
    MMP, LOCAL = __version__.split("+")
else:
    MMP = __version__
    LOCAL = ""
MAJOR, MINOR = [int(x) for x in MMP.split(".")[:2]]
PATCHLEVEL = MMP.split(".")[2]
__all__ = ["__version__", "MAJOR", "MINOR", "PATCHLEVEL", "LOCAL"]
