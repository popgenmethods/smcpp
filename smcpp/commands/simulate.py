import json
import os.path
import sys
from logging import getLogger

from . import command
from .. import model

logger = getLogger(__name__)

class Simulate(command.ConsoleCommand):
    "Simulate from a fitted model"
    def __init__(self, parser):
        super().__init__(parser)
        parser.add_argument('model', 
                metavar="model.final.json",
                help="fitted model.json file")
        parser.add_argument('n', help='diploid sample size', type=int)
        parser.add_argument('length', help='chromosome length (megabases)', type=float)
        parser.add_argument('output', metavar="out.vcf", help='output vcf')
        parser.add_argument('--contig_id', help="contig name in resulting VCF", default="1")
        parser.add_argument('-r', type=float, help="override per-generation recombination rate")
        parser.add_argument('-u', type=float, help="override per-generation mutation rate")

    def main(self, args):
        import msprime as msp
        d = json.load(open(args.model, "rt"))
        logger.debug("Import model:\n%s", d)
        model_cls = d['model']['class']
        m = getattr(model, model_cls).from_dict(d['model'])
        events = m.to_msp()
        args.r = args.r or d['rho'] / 2 / m.N0
        args.u = args.u or d['theta'] / 2 / m.N0
        pc = [msp.PopulationConfiguration(args.n) for _ in range(m.NPOP)]
        sim = msp.simulate(length=int(args.length * 1e6),
                           recombination_rate=args.r,
                           mutation_rate=args.u,
                           population_configurations=pc,
                           demographic_events=events)
        sim.write_vcf(open(args.output, "wt"),
                      ploidy=2, contig_id=args.contig_id)
