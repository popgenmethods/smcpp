from __future__ import absolute_import
from ._version import __version__
MMP, LOCAL = __version__.split("+")
MAJOR, MINOR, PATCHLEVEL = [int(x) for x in MMP.split(".")]
__all__ = ["__version__", "MAJOR", "MINOR", "PATCHLEVEL", "LOCAL"]
