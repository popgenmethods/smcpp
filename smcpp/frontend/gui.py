import os
import os.path
import sys

# Make stdout unbuffered
nonbuffered_stdout = os.fdopen(sys.stdout.fileno(), 'w', 0)
sys.stdout = nonbuffered_stdout

# Switch the flag to something a bit more rememberable
if "--console" in sys.argv:
    sys.argv[sys.argv.index("--console")] = "--ignore-gooey"

from gooey import Gooey, GooeyParser

from .common import init_subparsers

image_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'assets'))

@Gooey(progress_regex=r'.*EM iteration (\d+)/(\d+)',
        progress_expr='x[0] / x[1] * 100.',
        monospace_display=True,
        image_dir=image_dir)
def main():
    parser = GooeyParser()
    subparsers = parser.add_subparsers()
    init_subparsers(subparsers)
    args = parser.parse_args()
    args.func(args)
