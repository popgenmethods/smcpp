#!/usr/bin/env python2.7
'Perform various inference and data management tasks using SMC++'
from argparse import ArgumentParser
import sys
import os.path

import smcpp.commands as commands

# Simple wrapper class which ignores the widget option,
# enabling us to degrade gracefully in the non-GUI case.
class IgnorantArgumentParser(ArgumentParser):
    def add_argument(self, *args, **kwargs):
        kwargs.pop("widget", None)
        ArgumentParser.add_argument(self, *args, **kwargs)

def init_parser_class(parser_module, parser):
    parser_module.init_parser(parser)
    def main(args):
        parser_module.main(args)
    parser.set_defaults(func=main)

def run(parser, subparser_cls=None):
    subparsers = parser.add_subparsers(dest="subcommand", parser_class=subparser_cls)
    subparsers.required = True
    # Initialize arguments. Each object is responsible for setting the
    # args.func, where the work takes place.
    cmds = [
            ('estimate', commands.estimate, 'Fit SMC++ to data'),
            ('plot', commands.plot, 'Plot size history from fitted model'),
            ('posterior', commands.posterior, 'Posterior decoding for a region'),
            ('vcf2smc', commands.vcf2smc, 'Convert VCF to SMC++ format')
            ]
    for kwd, module, help in cmds:
        p = subparsers.add_parser(kwd, help=help)
        init_parser_class(module, p)
    # Go.
    args = parser.parse_args()
    args.func(args)

def console():
    run(ArgumentParser(), IgnorantArgumentParser)

def gui():
    run = Gooey(run)
    run(GooeyParser())
