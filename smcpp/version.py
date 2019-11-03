from pkg_resources import get_distribution, DistributionNotFound

try:
    version = get_distribution('smcpp').version
except DistributionNotFound:
    # package is not installed
    version = "ersion unknown"
