import inspect
import operator as op
import time
import timeit

N = 10000

import numpy as np


class Timer(object):
    """Context to compute time.
    """
    def __enter__(self):
        self.start = time.time()
        return self

    def __exit__(self, *args):
        self.end = time.time()
        self.secs = self.end - self.start
        self.msecs = self.secs * 1000  # millisecs

def time_func(func):
    """
    Time the execution of func() and return the 
    exectution time in msecs.
    """
    with Timer() as t:
        func()
    return t.msecs


class BenchNumpy(object):
    """
    This class wraps all the basic math and numpy implementation one can expect.
    The default dtype is np.float64.
    There are : 
         - unary_ops (like op.abs(x))
         - binary_ops (like op.add(x,y))
         - unary_ufuncs (like np.sqrt(x))
         - binary_ufuncs (like np.add(x,y))
    
    """
    # scalars
    scalars = [1.23]#, -1.1, -1.0, 0, 1,]
    # list of shapes to bench
    shapes = [(10,), (1000,), (100, 100), (2,)*10]
    # python's unary operators
    python_unary_ops = [
        op.abs, op.neg, op.pos,
    ]
    # python's binary operators 
    python_binary_ops = [
        op.add, op.sub, op.mul,
        op.floordiv, op.floordiv, op.truediv,
        op.mod, op.pow,
        op.lt, op.le, op.eq, op.ne, op.ge, op.gt
    ]
    # numpy's unary ufuncs 
    numpy_unary_ufuncs = [
        np.negative, np.absolute, np.rint, np.sign, np.conj, np.exp,
        np.exp2, np.log, np.log2, np.log10, np.expm1, np.log1p,
        np.sqrt, np.square, np.reciprocal, np.ones_like,
        np.sin, np.cos, np.tan, np.arcsin, np.arccos, np.arctan,
        np.sinh, np.cosh, np.tanh, np.arcsinh, np.arccosh, np.arctanh,
        np.deg2rad, np.rad2deg,
        np.floor, np.ceil, np.trunc,
    ]
    # numpy's binary ufuncs
    numpy_binary_ufuncs = [
        np.add, np.subtract, np.multiply, np.divide, np.logaddexp,
        np.logaddexp2, np.true_divide, np.floor_divide, np.power,
        np.remainder, np.mod, np.fmod,
        np.arctan2, np.hypot,
        np.greater, np.greater_equal, np.less, np.less_equal,
        np.not_equal, np.equal,
        np.maximum, np.minimum,
    ]
    # numpy's array functions
    numpy_array_function = [
        np.alen,  np.amax,  np.amin,  np.append,  np.argmax,  np.argmin,
        np.argsort,  np.around,  np.asanyarray,  np.atleast_1d,  np.atleast_2d,
        np.atleast_3d,  np.average,  np.broadcast_arrays,  np.broadcast_to,  np.clip,
        np.column_stack,  np.compress,  np.concatenate,  np.convolve,  np.copy,
        np.copyto,  np.corrcoef,  np.cov,  np.cross,  np.cumsum,  np.diagonal, 
        np.diff,  np.dot,  np.dstack,  np.empty_like,  np.expand_dims,  
        #np.fft, np.fft2,  np.fftn,  np.fftshift,  
        np.flip,  np.fliplr,  np.flipud,  
        np.full_like,  
        #np.hfft,
        np.histogram,  np.hstack,  
        #np.ifft,  np.ifft2, np.ifftn,  np.ifftshift,  np.ihfft,  
        np.interp,  
        np.linalg.inv,  
        #np.irfft, np.irfft2,  np.irfftn,  
        np.linspace,  
        np.linalg.lstsq,  
        np.may_share_memory, 
        np.mean,  np.median,  np.meshgrid,  np.ndim,  
        np.random.normal,
        np.ones_like, 
        np.percentile,  np.polyfit,  np.polyval,  np.prod,  np.ravel,  np.real,
        np.repeat,  np.reshape,  
        #np.rfft,  np.rfft2,  np.rfftn,  
        np.roll, np.rollaxis,  np.rot90,  np.searchsorted,  np.shape,  np.sort, 
        np.squeeze,  np.std,  np.sum,  np.take,  np.tile,  np.transpose,  np.trapz, 
        np.var,  np.vstack,  np.where,  np.zeros,  np.zeros_like,
    ]
        
    
    def __init__(self, dtype=np.float64):
        self.dtype = dtype
        
    
    @property
    def numerical_values_dict(self):
        return {
            "scalar":self.scalars,
            "array":self.set_arrays(),
        }
        
    @property
    def func_dict(self):
        return {
            "python_unary_ops":{
                "func_package":"python",
                "func_naryness":"unary",
                "func_list":self.python_unary_ops,
            },
            "python_binary_ops":{
                "func_package":"python",
                "func_naryness":"binary",
                "func_list":self.python_binary_ops,
            },
            "numpy_unary_ops":{
                "func_package":"numpy",
                "func_naryness":"unary",
                "func_list":self.numpy_unary_ufuncs,
            },
            "numpy_binary_ops":{
                "func_package":"numpy",
                "func_naryness":"binary",
                "func_list":self.numpy_binary_ufuncs,
            },
            "numpy_funcs":{
                "func_package":"numpy",
                "func_naryness":"unary",
                "func_list":self.numpy_array_function,
            },
        }
        
        
    def bench_all(self):
        rec_res_dict = []
        for _, func_dict in self.func_dict.items():
            func_package = func_dict["func_package"]
            func_naryness = func_dict["func_naryness"]
            func_list = func_dict["func_list"]
            
            # handle unary functions
            if func_naryness == "unary":
                for func in func_list:
                    for nv_kind, nv_list in self.numerical_values_dict.items():
                        for nv in nv_list:
                            # compute actual result and timeit if not failed
                            try:
                                res = func(nv)
                                passed = True
                                time = self.timeit(lambda :func(nv))
                            except Exception as e:
                                res = e
                                passed = False
                                time = None

                            rec_res_dict.append({
                                "package":func_package, 
                                "func":func,
                                "naryness":func_naryness,
                                "left":nv,
                                "type_left":type(nv),
                                "right":None,
                                "type_right":None,
                                "result":res,
                                "time":time,
                                "passed":passed,
                            })
            if func_naryness == "binary":
                for func in func_list:
                    for nv_kind, nv_list in self.numerical_values_dict.items():
                        for nv_left in nv_list:
                            for nv_right in nv_list:
                                # compute actual result and timeit if not failed
                                try:
                                    res = func(nv_left, nv_right)
                                    passed = True
                                    time = self.timeit(lambda :func(nv_left, nv_right))
                                except Exception as e:
                                    res = e
                                    passed = False
                                    time = None
    
                                rec_res_dict.append({
                                    "package":func_package, 
                                    "func":func,
                                    "naryness":func_naryness,
                                    "left":nv_left,
                                    "type_left":type(nv_left),
                                    "right":nv_right,
                                    "type_right":type(nv_right),
                                    "result":res,
                                    "time":time,
                                    "passed":passed,
                                })
        return rec_res_dict
                    
        

    def set_arrays(self):
        # create the arrays and store them
        return [self.rand(shape) for shape in self.shapes]

    def timeit(self, func):
        return timeit.timeit(func, number=N)
    
    #def bench_unary_funcs(self, funcs):
    #    rec_res_dict = []
    #    for f in funcs:
    #        for sca in self.scalars:
    #            
    #            # compute actual result and timeit if not failed
    #            try:
    #                res = f(sca)
    #                passed = True
    #                time = self.timeit(lambda :f(sca))
    #            except Exception as e:
    #                res = e
    #                passed = False
    #                time = None
    #            
    #            rec_res_dict.append({
    #                "package":"python", 
    #                "func":f,
    #                "func_kind":"python",
    #                "naryness":"unary",
    #                "left":sca,
    #                "type_left":"scalar",
    #                "right":"None",
    #                "result":res,
    #                "time":time,
    #                "passed":passed,
    #            })
    #        
    #        for arr in self.arrays:
    #            # compute actual result and timeit if not failed
    #            try:
    #                res = f(arr)
    #                passed = True
    #                time = self.timeit(lambda :f(arr))
    #            except Exception as e:
    #                res = e
    #                passed = False
    #                time = None
    #                
    #            rec_res_dict.append({
    #                "package":"python", 
    #                "func":f,
    #                "func_kind":"python",
    #                "naryness":"unary",
    #                "left":arr,
    #                "type_left":"array",
    #                "right":"None",
    #                "result":res,
    #                "time":time,
    #                "passed":passed
    #            })    
    #    return rec_res_dict
    

    def rand(self, shape):
        """
        Helper to generate an array of random numbers 
        between 0 and 10 with a given shape.
        """
        return (10 * np.random.rand(*shape)).astype(self.dtype, copy=False)

UNITS = ["m", "s", "J"]
    
class NewBenchModule(BenchNumpy):
    
    def __init__(self):
        self.qarrays = self.set_qarrays()
        self.qscalars = self.set_qscalars()
        super().__init__()
    
    # override numpy set_arrays with a make quantity
    def set_qarrays(self, units=UNITS):
        make_res = []
        for unit in units:
            for shape in self.shapes:
                try:
                    arr = self.rand(shape)
                    res = self.make(arr, unit)
                    time = self.timeit(lambda:self.make(arr, unit))
                    passed = True
                except Exception as e:
                    res = e
                    passed = False
                    time = None
                make_res.append({
                    "module":self.name,
                    "unit":unit,
                    "from":shape,
                    "passed":passed,
                    "result":res,
                    "time":time,
                })
        # set a list of array quantities
        self.qarrays = [d["result"] for d in make_res]
        return make_res
    
    def set_qscalars(self, units=UNITS):
        make_res = []
        for unit in units:
            for sca in self.scalars:
                try:
                    res = self.make(sca, unit)
                    time = self.timeit(lambda:self.make(sca, unit))
                    passed = True
                except Exception as e:
                    res = e
                    passed = False
                    time = None
                make_res.append({
                    "module":self.name,
                    "unit":unit,
                    "from":sca,
                    "passed":passed,
                    "result":res,
                    "time":time,
                })
        # set a list of scalar quantities
        self.qscalars = [d["result"] for d in make_res]
        return make_res        
    

class BenchModule(object):
    def __init__(self, np_obj):
        self.np_obj = np_obj
        self.unary_ops = [
            o for o in np_obj.unary_ops if self.test_unary(o)]
        self.binary_same_ops = [
            o for o in np_obj.binary_ops if self.test_binary_same(o)]
        self.binary_compatible_ops = [
            o for o in np_obj.binary_ops if self.test_binary_compatible(o)]
        self.binary_different_ops = [
            o for o in np_obj.binary_ops if self.test_binary_different(o)]
        self.unary_ufuncs = [
            o for o in np_obj.unary_ufuncs if self.test_unary(o)]
        self.binary_same_ufuncs = [
            o for o in np_obj.binary_ufuncs if self.test_binary_same(o)]
        self.binary_compatible_ufuncs = [
            o for o in np_obj.binary_ufuncs if self.test_binary_compatible(o)]
        self.binary_different_ufuncs = [
            o for o in np_obj.binary_ufuncs if self.test_binary_different(o)]
        self.other_numpy = self.test_other_numpy()

    ## Helpers
    def rand(self, shape):
        return self.np_obj.rand(shape)

    def test_unary(self, func):
        """"
        Create an array of 2 elements, compute the meter-array, and apply the unary func.
        """
        x = self.make(self.rand(shape=(2,)), units='m')
        try:
            func(x)
        except:
            return False
        return True

    def test_binary_same(self, func):
        """
        Create 2 arrays of 2 elements, compute the meter array of each, and apply binary func.
        """
        x = self.make(self.rand(shape=(2,)), units='m')
        y = self.make(self.rand(shape=(2,)), units='m')
        try:
            func(x, y)
        except:
            return False
        return True

    def test_binary_compatible(self, func):
        """
        Compatible means compatibility of dimension, like length + length.
        Create 2 arrays of 2 elements with same dimension
        (meter and feet or mile), and apply the binary func.
        """
        x = self.make(self.rand(shape=(2,)), units='m')
        try:
            y = self.make(self.rand(shape=(2,)), units='mile')
        except:
            try:
                y = self.make(self.rand(shape=(2,)), units='ft')
            except:
                return False
        try:
            func(x, y)
        except:
            return False
        return True

    def test_binary_different(self, func):
        """
        Different means different dimension, like time and length.
        Craete 2 arrays of 2 elements with different dimension
        (meter and second), and apply binary func.
        """
        x = self.make(self.rand(shape=(2,)), units='m')
        y = self.make(self.rand(shape=(2,)), units='s')
        try:
            func(x, y)
        except:
            return False
        return True

    def test_other_numpy(self):
        # For each, if it works we add it
        good = []
        bad = []
        q1 = self.make(self.rand((10,)), 'm')
        q2 = self.make(self.rand((5,)), 'm')

        try:
            np.where(q1 > self.make(0.0, 'm'))
        except Exception as e:
            bad.append(str(e))
        else:
            good.append(np.where)

        try:
            np.sort(q1)
        except Exception as e:
            bad.append(str(e))
        else:
            good.append(np.sort)

        try:
            np.argsort(q1)
        except Exception as e:
            bad.append(str(e))
        else:
            good.append(np.argsort)

        try:
            np.mean(q1)
        except Exception as e:
            bad.append(str(e))
        else:
            good.append(np.mean)

        try:
            np.std(q1)
        except Exception as e:
            bad.append(str(e))
        else:
            good.append(np.std)

        try:
            np.median(q1)
        except Exception as e:
            bad.append(str(e))
        else:
            good.append(np.median)

        try:
            np.concatenate((q1, q2))
        except Exception as e:
            bad.append(str(e))
        else:
            good.append(np.concatenate)

        return good

    def make_args(self, argspec, shape):
        np_args = []
        args = []
        for arg in argspec.args:
            if arg == 'self':
                continue

            if arg == 'shape':
                np_args.append(shape)
                args.append(shape)
                continue

            ndarray = self.rand(shape)
            if arg.startswith('neg'):
                ndarray *= -1
            np_args.append(ndarray)
            if arg.endswith('compatible'):
                try:
                    args.append(self.make(ndarray, 'mile'))
                except:
                    args.append(self.make(ndarray, 'ft'))
            elif arg.endswith('different'):
                args.append(self.make(ndarray, 's'))
            else:
                args.append(self.make(ndarray, 'm'))

        return tuple(np_args), tuple(args)

    def time_func(self, func, shapes=((1,), (1000,), (100, 100)),
                  iters=50, timeout=2000.0, verbose=False):
        np_time = []
        time = []
        argspec = inspect.getargspec(func)

        for i in range(iters):
            for shape in shapes:
                np_args, args = self.make_args(argspec, shape)
                try:
                    np_time.append(func(*np_args))
                except:
                    np_time.append(np.inf)
                try:
                    time.append(func(*args))
                except Exception as e:
                    return -1, -1, -1
                if time[-1] > timeout:
                    if verbose:
                        print("{}.{} timed out".format(self.name, func.__name__))
                    return 20.0, 20.0, 20.0

        # Get rid of the top and bottom 2
        if iters > 10:
            np_time.sort()
            np_time = np.asarray(np_time[2:-2])
            time.sort()
            time = np.asarray(time[2:-2])

        mean = np.mean(time)
        std = np.std(time)
        np_rel = np.sum(time) / np.sum(np_time)

        if verbose:
            print("{}.{}: {:.3f} +/- {:.2f} ms, {:.2f}x numpy".format(
                self.name, func.__name__, mean, std, np_rel))

        return mean, std, min(np_rel, 20.0)
    
    def time_make_arrq(self, arr, unit_str):
        return timeit.timeit(lambda:self.make(arr, unit_str), number=N)
    
    def time_make(self, shape):
        with Timer() as t:
            self.make(self.rand(shape), units='m')
        return t.msecs

    def time_ops(self, pos, neg_same, pos_compatible, neg_different):
        t = 0.0
        t += time_unary([pos, neg_same, pos_compatible, neg_different],
                        self.unary_ops)
        t += time_binary([pos], [neg_same], self.binary_same_ops)
        t += time_binary([pos], [neg_same, pos_compatible],
                         self.binary_compatible_ops)
        t += time_binary([pos], [neg_same, pos_compatible, neg_different],
                         self.binary_different_ops)
        return t

    def time_ufuncs(self, pos, neg_same, pos_compatible, neg_different):
        t = 0.0
        t += time_unary([pos, neg_same, pos_compatible, neg_different],
                        self.unary_ufuncs)
        t += time_binary([pos], [neg_same], self.binary_same_ufuncs)
        t += time_binary([pos], [neg_same, pos_compatible],
                         self.binary_compatible_ufuncs)
        t += time_binary([pos], [neg_same, pos_compatible, neg_different],
                         self.binary_different_ufuncs)
        return t

    ## Actual test functions that gather data

    def syntax(self, verbose=False):
        q = self.make(5.0, 'm')
        if verbose:
            print(q)
        return {
            'make': self.make_syntax,
            'str': str(q),
            'repr':repr(q),
        }

    def compatibility(self, verbose=False):
        compares = [
            # (BenchModule.attr, BenchNumPy.np_attr)
            ('unary_ops', 'unary_ops'),
            ('binary_same_ops', 'binary_ops'),
            ('binary_compatible_ops', 'binary_ops'),
            ('binary_different_ops', 'binary_ops'),
            ('unary_ufuncs', 'unary_ufuncs'),
            ('binary_same_ufuncs', 'binary_ufuncs'),
            ('binary_compatible_ufuncs', 'binary_ufuncs'),
            ('binary_different_ufuncs', 'binary_ufuncs'),
            ('other_numpy', 'other_numpy'),
        ]

        res = {}

        res['syntax'] = {}
        try:
            res['syntax']['print'] = str(self.make([5.0, 10.0], 'm'))
        except:
            res['syntax']['print'] = False
        try:
            res['syntax']['shape'] = str(self.make([5.0, 10.0], 'm').shape)
        except:
            res['syntax']['shape'] = False

        for attr, np_attr in compares:
            if verbose:
                self_n = len(getattr(self, attr))
                np_n = len(getattr(self.np_obj, np_attr))
                print("{}: {} / {} ({:.1%}".format(
                    attr, self_n, np_n, float(self_n) / np_n))

            res[attr] = {f.__name__: (f in getattr(self, attr))
                         for f in getattr(self.np_obj, np_attr)}

        return res

    def time(self, verbose=False):
        make_mean, make_std, make_rel = self.time_func(self.time_make)
        op_mean, op_std, op_rel = self.time_func(self.time_ops)
        ufunc_mean, ufunc_std, ufunc_rel = self.time_func(self.time_ufuncs)

        return {
            'make': {'mean': make_mean, 'std': make_std, 'np_rel': make_rel},
            'ops': {'mean': op_mean, 'std': op_std, 'np_rel': op_rel},
            'ufunc': {'mean': ufunc_mean, 'std': ufunc_std, 'np_rel': ufunc_rel},
        }

    ## To be overridden by subclasses

    @property
    def name(self):
        raise NotImplementedError()

    @property
    def make_syntax(self):
        raise NotImplementedError()

    def make(self, ndarray, units):
        raise NotImplementedError()


def bench(cls):
    """
    Main function to bench a package.
    
    cls is for example benchmarks.bench_astropy.BenchAstropy, and take an instance
    of a BenchNumpy.
    
    It has 2 main attributes : 
     - .name
     - .facts
    and 3 methods : 
     - .syntax()
     - .compatibility()
     - .time()
    
    """
    np_obj = BenchNumpy()
    b = cls(np_obj)

    res = {}
    res['name'] = b.name
    res['facts'] = b.facts
    try:
        res['syntax'] = b.syntax()
    except Exception as e:
        res['syntax'] = str(e)
    try:
        res['compatibility'] = b.compatibility()
    except Exception as e:
        res['compatibility'] = str(e)
    res['speed'] = b.time()
    return res


if __name__ == '__main__':
    import warnings
    warnings.simplefilter('ignore')
    np.seterr(all='ignore')
    bench = BenchNumpy()
    bench.time()

    