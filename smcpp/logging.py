from __future__ import absolute_import, division, print_function
import os
import logging
import multiprocessing

from logging import INFO, WARNING, DEBUG, NOTSET, CRITICAL

def init_logging(outdir, verbose, debug_log=os.devnull):
    logging.addLevelName(DEBUG - 1, 'DEBUG1')
    root = logging.getLogger()
    while len(root.handlers) > 0:
        root.removeHandler(logging.root.handlers[-1])
    fmt = logging.Formatter('%(relativeCreated)d %(name)-12s %(levelname)-8s %(message)s')
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    if verbose:
        sh.setLevel(DEBUG - verbose + 1)
    else:
        sh.setLevel(INFO)
    root.addHandler(sh)
    fh = logging.FileHandler(debug_log, "wt")
    fh.setLevel(DEBUG)
    fh.setFormatter(fmt)
    root.addHandler(fh)
    root.setLevel(NOTSET)

def getLogger(name):
    if multiprocessing.current_process().name == "MainProcess":
        return logging.getLogger(__name__)
    else:
        # return multiprocessing.get_logger()
        logger = multiprocessing.log_to_stderr()
        logger.setLevel(multiprocessing.SUBDEBUG)
        return logger
