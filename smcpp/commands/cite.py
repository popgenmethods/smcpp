from . import command

BIBTEX = """@article{smc++,
    Author = {Terhorst, Jonathan and Kamm, John A and Song, Yun S},
    Journal = {Nature Genetics},
    DOI = {10.1038/ng.3748},
    Number = {2},
    Pages = {303--309},
    Title = {Robust and scalable inference of population history 
             from hundreds of unphased whole genomes},
    Volume = {49},
    Year = {2017}
}
"""

PLAIN = """
J. Terhorst, J. A. Kamm, and Y. S. Song. Robust and scalable inference
of population history from hundreds of unphased whole genomes. Nature
Genetics, 49(2):303–309, 2017.
"""


class Cite(command.ConsoleCommand):
    'Print citation information for SMC++'

    def main(self, args):
        print(PLAIN)
        print("")
        print(BIBTEX)
