from __future__ import absolute_import
from ._version import __version__
if "+" in __version__:
    MMP, LOCAL = __version__.split("+")
else:
    MMP = __version__
    LOCAL = ""
MAJOR, MINOR, PATCHLEVEL = [int(x) for x in MMP.split(".")]
__all__ = ["__version__", "MAJOR", "MINOR", "PATCHLEVEL", "LOCAL"]
