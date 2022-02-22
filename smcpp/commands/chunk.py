import concurrent.futures
import numpy as np
from logging import getLogger

from . import command
from .. import data_filter, util

logger = getLogger(__name__)

class Chunk(command.Command, command.ConsoleCommand):
    "Sample randomly with replacement chunks from data file(s)"
    def __init__(self, parser):
        command.Command.__init__(self, parser)
        parser.add_argument("n", type=int, default=1,
                            help="Number of chunks to resample")
        parser.add_argument("chunk_size", type=int, default=int(5e6),
                            help="Size of each chunk")
        parser.add_argument("prefix", help="Prefix. Chunks will be sequentially numbered "
                            "<prefix>0.smc.gz, <prefix>1.smc.gz, etc.",
                            default="chunk")
        parser.add_argument('data', nargs="+",
                            help="data file(s) in SMC++ format")

    def main(self, args):
        command.Command.main(self, args)
        with util.optional_gzip(args.data[0], "rt") as f:
            header = next(f).strip()
        logger.debug(header)
        pipe = self._pipeline = data_filter.DataPipeline(args.data)
        pipe.add_filter(load_data=data_filter.LoadData())
        pipe.add_filter(chunk=data_filter.Chunk(args.chunk_size))
        chunks = [chunk for chunks in pipe.results() for chunk in chunks]
        np.random.seed(args.seed)
        spls = np.random.choice(chunks, size=args.n, replace=True)
        fns = [args.prefix + str(i) + '.smc.gz' for i in range(args.n)]
        with concurrent.futures.ProcessPoolExecutor() as pool:
            list(pool.map(_chunk_helper, 
                fns,
                spls, 
                ["%d"] * args.n,
                [header] * args.n
                ))
        logger.info("Wrote file(s): %s, ..., %s", fns[0], fns[-1])


def _chunk_helper(fn, X, fmt, hdr):
    np.savetxt(fname=fn, X=X, fmt=fmt, header=hdr, comments="")
