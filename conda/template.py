#!/usr/bin/env python

tpl, ver = sys.argv[1:]
print(open(tpl, "rt").read().format(version=ver[1:]))
