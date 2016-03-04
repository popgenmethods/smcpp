import numpy as np
from smcpp._smcpp import sfs, T_MAX
n = 10 # number of undistinguished haploids
N0 = 10000.

# Simple demographic model. (haploid) population size grows
# exponentially from 10000 to 200000 over the last 10000 years.
# (constant before that)
model = np.array([[2e5, 1e4], [1e4, 1e4], [1e4, 1e6]]) / (2. * N0)
model[2] /= 25. # Assume generation time of 25 years

# Population scaled mutation rate
theta = 1.2e-8 * 2. * N0

# Conditional interval for distinguished lineage coalescence
t0, t1 = (0., T_MAX) # think of T_MAX as infinity. :)

# False at last parameter indicates that gradient information should not
# be returned.
table = sfs(model, n, t0, t1, theta, False) 

print(table)
