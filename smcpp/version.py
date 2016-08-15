from setuptools_scm import get_version

__version__ = get_version()
if "+" in __version__:
    MMP, LOCAL = __version__.split("+")
else:
    MMP = __version__
    LOCAL = ""
MAJOR, MINOR, PATCHLEVEL = [int(x) for x in MMP.split(".")[:3]]
__all__ = ["__version__", "MAJOR", "MINOR", "PATCHLEVEL", "LOCAL"]
