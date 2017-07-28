from __future__ import absolute_import, division, print_function
import os
import logging
import multiprocessing
import wrapt

from logging import INFO, ERROR, WARNING, DEBUG, NOTSET, CRITICAL


class _SMCPPFilter:
    def filter(self, record):
        return record.name.startswith("smcpp")


def init_logging():
    # Get rid of any pre-existing stuff
    root = logging.getLogger()
    while len(root.handlers) > 0:
        root.removeHandler(logging.root.handlers[-1])
    logging.addLevelName(logging.DEBUG - 1, 'DEBUG1')
    fmt = logging.Formatter(
        '%(relativeCreated)d %(name)-12s %(levelname)-1s %(message)s')
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    sh.setLevel(logging.INFO)
    sh.addFilter(_SMCPPFilter())
    root.addHandler(sh)
    root.setLevel(logging.NOTSET)


def getLogger(name):
    return logging.getLogger(name)


def setup_logging(verbosity):
    root = logging.getLogger()
    sh = root.handlers[0]
    sh.setLevel([INFO, DEBUG, DEBUG - 1][verbosity])
    logging.captureWarnings(True)

def add_debug_log(debug_log):
    fh = logging.FileHandler(debug_log, "wt")
    fh.setLevel(DEBUG)
    root = logging.getLogger()
    sh = root.handlers[0]
    fh.setFormatter(sh.formatter)
    root.addHandler(fh)
