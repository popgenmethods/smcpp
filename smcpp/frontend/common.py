import sys
import os.path

from .. import commands

def init_parser_class(parser_module, parser):
    parser_module.init_parser(parser)
    def main(args):
        parser_module.main(args)
    parser.set_defaults(func=main)

CMDS = [
        ('estimate', commands.estimate, 'Fit SMC++ to data'),
        ('split', commands.split, 'Estimate split time in two population model'),
        ('plot', commands.plot, 'Plot size history from fitted model'),
        ('posterior', commands.posterior, 'Plot posterior decoding for a region'),
        ('vcf2smc', commands.vcf2smc, 'Convert VCF to SMC++ format'),
        ]

def init_subparsers(subparsers_obj):
    for kwd, module, help in CMDS:
        p = subparsers_obj.add_parser(kwd, help=help)
        init_parser_class(module, p)
