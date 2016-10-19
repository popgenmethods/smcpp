from argparse import ArgumentParser

from .common import init_subparsers, CMDS

def main():
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')
    subparsers.required = True
    init_subparsers(subparsers)
    args = parser.parse_args()
    CMDS[args.command][0].main(args)
