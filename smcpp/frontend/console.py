from argparse import ArgumentParser

from .common import init_subparsers

# Simple wrapper class which ignores the widget option,
# enabling us to degrade gracefully in the non-GUI case.
class IgnorantArgumentParser(ArgumentParser):
    def add_argument(self, *args, **kwargs):
        kwargs.pop("widget", None)
        ArgumentParser.add_argument(self, *args, **kwargs)

def main():
    parser = ArgumentParser()
    subparsers = parser.add_subparsers(parser_class=IgnorantArgumentParser)
    init_subparsers(subparsers)
    args = parser.parse_args()
    args.func(args)
