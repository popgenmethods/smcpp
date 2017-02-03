from ..version import version

from . import command

class Version(command.ConsoleCommand):
    'Print version string'

    def main(self, args):
        print("SMC++ v%s" % version)
