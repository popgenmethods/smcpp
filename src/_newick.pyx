def tmrca(newick, l1, l2):
    return cython_tmrca(newick, l1, l2) / 2.0
