from __future__ import absolute_import, division, print_function
import os
import logging
import multiprocessing
import wrapt

from logging import INFO, WARNING, DEBUG, NOTSET, CRITICAL

def getLogger(name):
    if multiprocessing.current_process().name == "MainProcess":
        return logging.getLogger(name)
    else:
        return multiprocessing.get_logger()

def setup_logging(verbosity, debug_log=None):
    root = logging.getLogger()
    sh = root.handlers[0]
    sh.setLevel([INFO, DEBUG, DEBUG - 1][verbosity])
    if debug_log is not None:
        fh = logging.FileHandler(debug_log, "wt")
        fh.setLevel(DEBUG)
        fh.setFormatter(sh.formatter)
        root.addHandler(fh)

def log_step(entrance_msg, exit_msg):
    @wrapt.decorator
    def wrapper(wrapped, instance, args, kwargs):
        logger = wrapped.__globals__['logger']
        logger.info(entrance_msg)
        ret = wrapped(*args, **kwargs)
        logger.info(exit_msg)
        return ret
    return wrapper
