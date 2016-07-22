def tmrca(newick, l1, l2):
    newick, l1, l2 = [s.encode() for s in (newick, l1, l2)]
    return cython_tmrca(newick, l1, l2) / 2.0
