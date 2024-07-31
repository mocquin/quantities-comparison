import numpy as np
import physipy

from . import base


class BenchPhysipy(base.NewBenchModule):
    facts = {
        'LOC': 0,
        'First release': '20XX-XX',
        'Most recent release': '20XX-XX',
        'Implementation': 'Container?',
        'URL': '',
        'PyPI': 'physipy',
    }

    def __init__(self):
        self.units = physipy.units
        self.units["ft"] = physipy.imperial_units["ft"]
        super().__init__()

    @property
    def name(self):
        return physipy.__name__

    @property
    def make_syntax(self):
        return "multiply"

    def make(self, ndarray, units):
        return ndarray * self.units[units]


if __name__ == '__main__':
    import warnings
    warnings.simplefilter('ignore')
    np.seterr(all='ignore')
    base.bench(BenchPhysipy)
