from . import command

BIBTEX = """@article{terhorst2016,
    Author = {Terhorst, Jonathan and Kamm, John A and Song, Yun S},
    Journal = {Nature Genetics},
    Title = {Robust and scalable inference of population history from
             hundreds of unphased whole genomes},
    Year = {in press}
}
"""

PLAIN = """J. Terhorst, J. A. Kamm, and Y. S. Song. Robust and scalable inference
of population history from hundreds of unphased whole genomes. Nature
Genetics, in press.
"""


class Cite(command.Command):
    'Print citation information for SMC++'

    def main(self, args):
        print("")
        print(PLAIN)
        print("")
        print(BIBTEX)
