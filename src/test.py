import mpmath.libmp
import sys

x = mpmath.libmp.from_str(sys.argv[1], 100)
print mpmath.libmp.to_str(mpmath.libmp.mpf_ei(x, 100), 100)
