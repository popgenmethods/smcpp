import multiprocessing

from . import logging

logger = logging.getLogger(__name__)

# Due to bugs in Python and/or OpenMP and/or GCC (?) it is essential to
# create workers before *any* OpenMP threads get created.

_pool = None

def init_pool():
    global _pool
    if multiprocessing.current_process().name == "MainProcess" and _pool is None:
        logger.info("initializing pool")
        _pool = multiprocessing.Pool()

def get_pool():
    assert multiprocessing.current_process().name == "MainProcess"
    return _pool
