#!/usr/bin/env python2.7
'Perform various inference and data management tasks using SMC++'
import sys
import os.path

import smcpp.commands as commands

def init_parser_class(parser_module, parser):
    parser_module.init_parser(parser)
    def main(args):
        parser_module.main(args)
    parser.set_defaults(func=main)

CMDS = [
        ('estimate', commands.estimate, 'Fit SMC++ to data'),
        ('plot', commands.plot, 'Plot size history from fitted model'),
        ('posterior', commands.posterior, 'Plot posterior decoding for a region'),
        ('vcf2smc', commands.vcf2smc, 'Convert VCF to SMC++ format')
        ]

def init_subparsers(subparsers_obj):
    for kwd, module, help in CMDS:
        p = subparsers_obj.add_parser(kwd, help=help)
        init_parser_class(module, p)
