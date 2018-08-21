#!/usr/bin/env python3

import msprime as msp
from smcpp.util import sawtooth
import sys

if __name__ == "__main__":
    sawtooth_events = [
        (.000582262, 1318.18),
        (.00232905, -329.546),
        (.00931919, 82.3865),
        (.0372648, -20.5966),
        (.149059, 5.14916),
        (0.596236, 0.),
    ]
    N0 = 14312
    de = (
            [msp.PopulationParametersChange(time=0, initial_size=5 * N0)] + 
            [
                msp.PopulationParametersChange(
                time=4 * t * N0,
                growth_rate=g / (4 * N0))
                for t, g in sawtooth_events
                ])
    msp.DemographyDebugger(demographic_events=de).print_history()
    msp.simulate(int(sys.argv[1]), length=1e8, mutation_rate=1.25e-8, recombination_rate=1e-9, 
            demographic_events=de).write_vcf(open(sys.argv[2], "wt"), ploidy=2)
