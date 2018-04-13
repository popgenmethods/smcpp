#!/usr/bin/env python
import msprime as msp
import sys

de = [msp.MassMigration(time=5000,
                            source=1, 
                            destination=0, 
                            proportion=1.0)]
pc = [msp.PopulationConfiguration(sample_size = 20) for _ in range(2)]

tree_seq = msp.simulate(population_configurations=pc, 
                        demographic_events=de,
                        recombination_rate=1e-9,
                        mutation_rate=1e-8,
                        length=int(sys.argv[1]),
                        Ne=50000)
tree_seq.write_vcf(sys.stdout, ploidy=2)
