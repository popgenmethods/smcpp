from __future__ import absolute_import, division, print_function
import os
import logging
import multiprocessing

from logging import INFO, WARNING, DEBUG, NOTSET, CRITICAL

def getLogger(name):
    if multiprocessing.current_process().name == "MainProcess":
        return logging.getLogger(__name__)
    else:
        return multiprocessing.get_logger()

def setup_logging(verbosity, debug_log):
    root = logging.getLogger()
    sh = root.handlers[0]
    sh.setLevel([INFO, DEBUG, DEBUG - 1][verbosity])
    fh = logging.FileHandler(debug_log, "wt")
    fh.setLevel(DEBUG)
    fh.setFormatter(sh.formatter)
    root.addHandler(fh)
