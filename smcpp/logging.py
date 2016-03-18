def init_logging(outdir, verbose, debug_log=os.devnull):
    import logging
    logging.addLevelName(logging.DEBUG-1, 'DEBUG1')
    root = logging.getLogger()
    while len(root.handlers) > 0:
        root.removeHandler(logging.root.handlers[-1])
    fmt = logging.Formatter('%(relativeCreated)d %(name)-12s %(levelname)-8s %(message)s')
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    if verbose:
        sh.setLevel(logging.DEBUG - 1)
    else:
        sh.setLevel(logging.INFO)
    root.addHandler(sh)
    fh = logging.FileHandler(debug_log, "wt")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    root.addHandler(fh)
    root.setLevel(logging.NOTSET)

