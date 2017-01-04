from argparse import ArgumentParser

from .. import commands


def init_subparsers(subparsers_obj):
    ret = {}
    kwds = {cls.__name__.lower(): cls
            for cls in commands.command.Command.__subclasses__()}
    for kwd in sorted(kwds):
        cls = kwds[kwd]
        p = subparsers_obj.add_parser(kwd, help=cls.__doc__)
        ret[kwd] = cls(p)
    return ret


def main():
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(dest='command')
    subparsers.required = True
    cmds = init_subparsers(subparsers)
    args = parser.parse_args()
    cmds[args.command].main(args)
