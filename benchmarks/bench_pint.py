import numpy as np
import pint

from . import base


class BenchPint(base.NewBenchModule):
    facts = {
        'LOC': 2914,
        'First release': '2012-07',
        'Most recent release': '2014-02',
        'Implementation': 'Container',
        'URL': 'https://pint.readthedocs.org/en/latest/',
        'PyPI': 'pint',
    }

    def __init__(self, ):
        self.unitreg = pint.UnitRegistry()
        self.m = self.unitreg.m
        self.s = self.unitreg.s
        self.J = self.unitreg.J
        
        super().__init__()
        
    @property
    def name(self):
        return pint.__name__

    @property
    def make_syntax(self):
        return "multiply"

    def make(self, ndarray, units):
        return ndarray * getattr(self.unitreg, units)


if __name__ == '__main__':
    import warnings
    warnings.simplefilter('ignore')
    np.seterr(all='ignore')
    base.bench(BenchPint)
