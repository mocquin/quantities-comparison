import numpy as np
import forallpeople

from . import base


class BenchForallpeople(base.BenchModule):
    facts = {
        'LOC': 0,
        'First release': '20XX-XX',
        'Most recent release': '20XX-XX',
        'Implementation': 'Container?',
        'URL': '',
        'PyPI': 'forallpeople',
    }

    def __init__(self, np_obj):
        ft = 0.3048*forallpeople.m
        mile= 1609.344*forallpeople.m
        self.units = {"m":forallpeople.m,
                     "s":forallpeople.s,
                      "ft":ft,
                      "mile":mile,
                     }
        #self.units["ft"] = physipy.imperial_units["ft"]
        base.BenchModule.__init__(self, np_obj)

    @property
    def name(self):
        return forallpeople.__name__

    @property
    def make_syntax(self):
        return "multiply"

    def make(self, ndarray, units):
        return ndarray * self.units[units]


if __name__ == '__main__':
    import warnings
    warnings.simplefilter('ignore')
    np.seterr(all='ignore')
    base.bench(BenchPint)
