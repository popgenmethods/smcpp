from __future__ import absolute_import, division, print_function
import os
import logging
import multiprocessing
import wrapt

from logging import INFO, ERROR, WARNING, DEBUG, NOTSET, CRITICAL

def getLogger(name):
    return logging.getLogger(name)
    # if multiprocessing.current_process().name == "MainProcess":
    # else:
        # return multiprocessing.get_logger()

def setup_logging(verbosity):
    root = logging.getLogger()
    sh = root.handlers[0]
    sh.setLevel([INFO, DEBUG, DEBUG - 1][verbosity])

def add_debug_log(debug_log):
    fh = logging.FileHandler(debug_log, "wt")
    fh.setLevel(DEBUG)
    root = logging.getLogger()
    sh = root.handlers[0]
    fh.setFormatter(sh.formatter)
    root.addHandler(fh)
