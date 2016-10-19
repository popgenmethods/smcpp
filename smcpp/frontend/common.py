import sys
import os.path

from .. import commands

CMDS = {
        'estimate': (commands.estimate, 'Fit SMC++ to data'),
        'split': (commands.split, 'Estimate split time in two population model'),
        'plot': (commands.plot, 'Plot size history from fitted model'),
        'vcf2smc': (commands.vcf2smc, 'Convert VCF to SMC++ format'),
        }

def init_subparsers(subparsers_obj):
    for kwd in sorted(CMDS):
        module, help = CMDS[kwd]
        p = subparsers_obj.add_parser(kwd, help=help)
        module.init_parser(p)
