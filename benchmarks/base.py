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
# Python Unary Operations
python_unary_ops = [
    op.abs,   # abs(x) -> absolute value of x
    op.inv,   # inv(x) -> inverse of x
    op.neg,   # neg(x) -> negation of x
    op.pos    # pos(x) -> identity function of x
]

# Python Binary Operations
python_binary_ops = [
    op.add,      # add(x1, x2) -> sum of x1 and x2
    op.eq,       # eq(x1, x2) -> element-wise equality comparison
    op.floordiv, # floordiv(x1, x2) -> integer division of x1 by x2
    op.ge,       # ge(x1, x2) -> element-wise greater than or equal to comparison
    op.gt,       # gt(x1, x2) -> element-wise greater than comparison
    op.lt,       # lt(x1, x2) -> element-wise less than comparison
    op.le,       # le(x1, x2) -> element-wise less than or equal to comparison
    op.mod,      # mod(x1, x2) -> element-wise modulus of x1 by x2
    op.mul,      # mul(x1, x2) -> element-wise multiplication of x1 and x2
    op.ne,       # ne(x1, x2) -> element-wise not equal to comparison
    op.pow,      # pow(x1, x2) -> element-wise x1 raised to the power of x2
    op.sub,      # sub(x1, x2) -> element-wise subtraction of x2 from x1
    op.truediv   # truediv(x1, x2) -> element-wise division of x1 by x2
]

# NumPy Unary Ufuncs
numpy_unary_ufuncs = [
    np.abs,        # abs(x) -> absolute value of x
    np.arccos,     # arccos(x) -> inverse cosine of x
    np.arcsinh,    # arcsinh(x) -> inverse hyperbolic sine of x
    np.arccosh,    # arccosh(x) -> inverse hyperbolic cosine of x
    np.arctan,     # arctan(x) -> inverse tangent of x
    np.arctanh,    # arctanh(x) -> inverse hyperbolic tangent of x
    np.ceil,       # ceil(x) -> ceiling of x
    np.conj,       # conj(x) -> complex conjugate of x
    np.cos,        # cos(x) -> cosine of x
    np.deg2rad,    # deg2rad(x) -> convert degrees to radians
    np.exp,        # exp(x) -> exponential of x
    np.exp2,       # exp2(x) -> 2**x
    np.expm1,      # expm1(x) -> exp(x) - 1
    np.floor,      # floor(x) -> floor of x
    np.log,        # log(x) -> natural logarithm of x
    np.log1p,      # log1p(x) -> log(1 + x)
    np.log10,      # log10(x) -> base-10 logarithm of x
    np.log2,       # log2(x) -> base-2 logarithm of x
    np.negative,   # negative(x) -> negation of x
    np.rad2deg,    # rad2deg(x) -> convert radians to degrees
    np.reciprocal, # reciprocal(x) -> reciprocal of x
    np.rint,       # rint(x) -> round to the nearest integer
    np.sqrt,       # sqrt(x) -> square root of x
    np.square,     # square(x) -> square of x
    np.sin,        # sin(x) -> sine of x
    np.sinh,       # sinh(x) -> hyperbolic sine of x
    np.tanh,       # tanh(x) -> hyperbolic tangent of x
    np.trunc       # trunc(x) -> truncate towards zero
]

# NumPy Binary Ufuncs
numpy_binary_ufuncs = [
    np.add,         # add(x1, x2) -> sum of x1 and x2
    np.arctan2,     # arctan2(x1, x2) -> element-wise arctangent of x1/x2
    np.divide,      # divide(x1, x2) -> element-wise division of x1 by x2
    np.fmax,        # fmax(x1, x2) -> element-wise maximum of x1 and x2
    np.fmin,        # fmin(x1, x2) -> element-wise minimum of x1 and x2
    np.hypot,       # hypot(x1, x2) -> element-wise hypotenuse of x1 and x2
    np.less,        # less(x1, x2) -> element-wise less than comparison
    np.less_equal,  # less_equal(x1, x2) -> element-wise less than or equal to comparison
    np.maximum,     # maximum(x1, x2) -> element-wise maximum of x1 and x2
    np.mod,         # mod(x1, x2) -> element-wise modulus of x1 by x2
    np.multiply,    # multiply(x1, x2) -> element-wise multiplication of x1 and x2
    np.not_equal,   # not_equal(x1, x2) -> element-wise not equal to comparison
    np.power,       # power(x1, x2) -> element-wise x1 raised to the power of x2
    np.remainder,   # remainder(x1, x2) -> element-wise remainder of x1 divided by x2
    np.subtract,    # subtract(x1, x2) -> element-wise subtraction of x2 from x1
    np.true_divide  # true_divide(x1, x2) -> element-wise division of x1 by x2 (true division)
]

# NumPy Array Functions
numpy_array_function = [
    np.amax,            # amax(a, axis=None, out=None, keepdims=False) -> maximum of array elements
    np.amin,            # amin(a, axis=None, out=None, keepdims=False) -> minimum of array elements
    np.append,          # append(arr, values, axis=None) -> append values to the end of an array
    np.argmax,          # argmax(a, axis=None, out=None) -> indices of the maximum values along an axis
    np.argmin,          # argmin(a, axis=None, out=None) -> indices of the minimum values along an axis
    np.argsort,         # argsort(a, axis=-1, kind='quicksort', order=None) -> indices that would sort an array
    np.around,          # around(a, decimals=0, out=None) -> round to the nearest integer
    np.asanyarray,      # asanyarray(a, dtype=None, order=None) -> convert input to an ndarray
    np.atleast_1d,     # atleast_1d(*arys) -> view inputs as at least one-dimensional arrays
    np.atleast_2d,     # atleast_2d(*arys) -> view inputs as at least two-dimensional arrays
    np.atleast_3d,     # atleast_3d(*arys) -> view inputs as at least three-dimensional arrays
    np.average,        # average(a, axis=None, weights=None, returned=False) -> compute the weighted average
    np.broadcast_arrays, # broadcast_arrays(*args, subok=False) -> broadcast arrays against each other
    np.broadcast_to,   # broadcast_to(array, shape, subok=False) -> broadcast an array to a given shape
    np.clip,           # clip(a, a_min, a_max, out=None) -> constrain values to within a given range
    np.column_stack,   # column_stack(tup) -> stack 1-D arrays as columns into a 2-D array
    np.compress,       # compress(condition, a, axis=None, out=None) -> select elements of an array based on condition
    np.concatenate,    # concatenate(arrays, axis=0, out=None) -> join a sequence of arrays along an existing axis
    np.convolve,       # convolve(a, v, mode='full') -> compute the convolution of two arrays
    np.copy,           # copy(a, order='K') -> return a copy of the array
    np.copyto,         # copyto(dest, src, casting='same_kind', where=True) -> copy values from src to dest
    np.corrcoef,       # corrcoef(x, y=None, rowvar=True, bias=False, ddof=None) -> correlation coefficient matrix
    np.cov,            # cov(m, y=None, rowvar=True, bias=False, ddof=None) -> covariance matrix of the variables
    np.cross,          # cross(a, b, axisa=0, axisb=0, axisc=0, out=None) -> cross product of two arrays
    np.cumsum,         # cumsum(a, axis=None, dtype=None, out=None) -> cumulative sum of the elements along an axis
    np.diagonal,       # diagonal(a, offset=0, axis1=0, axis2=1) -> extract a diagonal or diagonals from an array
    np.diff,           # diff(a, n=1, axis=-1) -> compute the n-th discrete difference along a given axis
    np.dot,            # dot(a, b, out=None) -> dot product of two arrays
    np.dstack,         # dstack(tup) -> stack arrays in sequence depth wise
    np.empty_like,     # empty_like(prototype, dtype=None, order='K', subok=True, shape=None) -> create a new array with the same shape and type as a given array
    np.expand_dims,    # expand_dims(a, axis) -> expand the shape of an array
    np.flip,           # flip(m, axis=None) -> reverse the order of elements in an array along the specified axis
    np.fliplr,         # fliplr(m) -> reverse the order of elements along the second axis
    np.flipud,         # flipud(m) -> reverse the order of elements along the first axis
    np.full_like,      # full_like(prototype, fill_value, dtype=None, shape=None, order='K') -> create a new array with the same shape and type as a given array, filled with fill_value
    np.histogram,      # histogram(a, bins=10, range=None, density=False, weights=None, return_edges=False) -> compute the histogram of a set of data
    np.hstack,         # hstack(tup) -> stack arrays in sequence horizontally (column wise)
    np.interp,         # interp(x, xp, fp, left=None, right=None, period=None) -> one-dimensional linear interpolation
    np.linspace,       # linspace(start, stop, num=50, endpoint=True, retstep=False, dtype=None, axis=0) -> create an array with evenly spaced values
    np.may_share_memory, # may_share_memory(a, b) -> check whether two arrays might share memory
    np.mean,           # mean(a, axis=None, dtype=None, out=None, keepdims=False) -> compute the arithmetic mean
    np.median,         # median(a, axis=None, out=None, keepdims=False) -> compute the median of the data
    np.meshgrid,       # meshgrid(*xi, indexing='xy') -> return coordinate matrices from coordinate vectors
    np.ndim,           # ndim(a) -> return the number of dimensions of an array
    np.ones_like,      # ones_like(prototype, dtype=None, shape=None, order='K') -> create a new array with the same shape and type as a given array, filled with ones
    np.percentile,     # percentile(a, q, axis=None, out=None, keepdims=False) -> compute the q-th percentile of the data
    np.polyfit,        # polyfit(x, y, deg, rcond=None, full=False, w=None, cov=False) -> least-squares polynomial fit
    np.polyval,        # polyval(p, x) -> evaluate a polynomial at specific values
    np.prod,           # prod(a, axis=None, dtype=None, out=None, keepdims=False) -> compute the product of array elements
    np.ravel,          # ravel(a, order='C') -> return a contiguous flattened array
    np.real,           # real(x) -> return the real part of the complex data
    np.repeat,         # repeat(a, repeats, axis=None) -> repeat elements of an array
    np.reshape,        # reshape(a, newshape, order='C') -> give a new shape to an array without changing its data
    np.roll,           # roll(a, shift, axis=None) -> roll array elements along a given axis
    np.rollaxis,       # rollaxis(a, axis, start=0) -> roll the specified axis backwards
    np.rot90,          # rot90(m, k=1, axes=(0, 1)) -> rotate an array by 90 degrees in the plane specified by axes
    np.searchsorted,   # searchsorted(a, v, side='left', sorter=None) -> find indices where elements should be inserted to maintain order
    np.shape,          # shape(a) -> return the shape of an array
    np.sort,           # sort(a, axis=-1, kind='quicksort', order=None) -> return a sorted copy of an array
    np.squeeze,        # squeeze(a, axis=None) -> remove single-dimensional entries from the shape of an array
    np.std,            # std(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False) -> compute the standard deviation
    np.sum,            # sum(a, axis=None, dtype=None, out=None, keepdims=False) -> compute the sum of array elements
    np.take,           # take(a, indices, axis=None, out=None, mode='raise') -> take elements from an array along an axis
    np.tile,           # tile(a, reps) -> construct an array by repeating a given array
    np.transpose,      # transpose(a, axes=None) -> permute the dimensions of an array
    np.trapz,          # trapz(y, x=None, dx=1.0, axis=-1) -> integrate along the given axis using the composite trapezoidal rule
    np.var,            # var(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False) -> compute the variance of array elements
    np.vstack,         # vstack(tup) -> stack arrays in sequence vertically (row wise)
    np.where,          # where(condition, x, y) -> return elements chosen from x or y depending on condition
    np.zeros,          # zeros(shape, dtype=float, order='C') -> return a new array of given shape and type, filled with zeros
    np.zeros_like      # zeros_like(prototype, dtype=None, shape=None, order='K') -> create a new array with the same shape and type as a given array, filled with zeros
]

# NumPy Linear Algebra Unary Ufuncs
numpy_linalg_unary_ufuncs = [
    np.linalg.cholesky, # cholesky(a) -> Cholesky decomposition of a matrix
    np.linalg.det,      # det(a) -> compute the determinant of an array
    np.linalg.eig,      # eig(a) -> compute the eigenvalues and right eigenvectors of a square array
    np.linalg.eigvals,  # eigvals(a) -> compute the eigenvalues of a square array
    np.linalg.eigh,     # eigh(a, UPLO='L') -> compute the eigenvalues and eigenvectors of a Hermitian or symmetric matrix
    np.linalg.inv,      # inv(a) -> compute the (multiplicative) inverse of a matrix
    np.linalg.matrix_rank, # matrix_rank(a, tol=None) -> return the matrix rank of array a
    np.linalg.pinv,     # pinv(a, rcond=1e-15) -> compute the (Moore-Penrose) pseudo-inverse of a matrix
    np.linalg.qr,       # qr(a, mode='reduced') -> compute the QR decomposition of a matrix
    np.linalg.svd       # svd(a, full_matrices=True, compute_uv=True) -> compute the singular value decomposition of a matrix
]

# NumPy Linear Algebra Binary Ufuncs
numpy_linalg_binary_ufuncs = [
    np.linalg.lstsq, # lstsq(a, b, rcond='warn') -> solve the least-squares problem for a linear system
    np.linalg.multi_dot # multi_dot(arrays) -> compute the dot product of multiple arrays
]

# NumPy Random Unary Ufuncs
numpy_random_unary_ufuncs = [
    np.random.beta,            # beta(a, b, size=None) -> draw samples from a Beta distribution
    np.random.binomial,        # binomial(n, p, size=None) -> draw samples from a Binomial distribution
    np.random.choice,          # choice(a, size=None, replace=True, p=None) -> generate random samples from a given 1-D array
    np.random.chisquare,       # chisquare(df, size=None) -> draw samples from a Chi-square distribution
    np.random.exponential,     # exponential(scale=1.0, size=None) -> draw samples from an Exponential distribution
    np.random.gamma,           # gamma(shape, scale=1.0, size=None) -> draw samples from a Gamma distribution
    np.random.geometric,       # geometric(p, size=None) -> draw samples from a Geometric distribution
    np.random.gumbel,          # gumbel(loc=0.0, scale=1.0, size=None) -> draw samples from a Gumbel distribution
    np.random.randint,         # randint(low, high=None, size=None, dtype=int) -> generate random integers from the discrete uniform distribution
    np.random.logistic,        # logistic(loc=0.0, scale=1.0, size=None) -> draw samples from a Logistic distribution
    np.random.lognormal,       # lognormal(mean=0.0, sigma=1.0, size=None) -> draw samples from a Log-normal distribution
    np.random.multinomial,     # multinomial(n, pvals, size=None) -> draw samples from a Multinomial distribution
    np.random.multivariate_normal, # multivariate_normal(mean, cov, size=None) -> draw samples from a Multivariate Normal distribution
    np.random.normal,          # normal(loc=0.0, scale=1.0, size=None) -> draw samples from a Normal distribution
    np.random.poisson,         # poisson(lam=1.0, size=None) -> draw samples from a Poisson distribution
    np.random.power,           # power(a, size=None) -> draw samples from a Power distribution
    np.random.rand,            # rand(d0, d1, ..., dn) -> generate random numbers from a uniform distribution over [0, 1)
    np.random.random,         # random(size=None) -> generate random numbers from a uniform distribution over [0, 1)
    np.random.standard_cauchy, # standard_cauchy(size=None) -> draw samples from a Standard Cauchy distribution
    np.random.standard_exponential, # standard_exponential(size=None) -> draw samples from a Standard Exponential distribution
    np.random.standard_gamma, # standard_gamma(shape, size=None) -> draw samples from a Standard Gamma distribution
    np.random.standard_normal, # standard_normal(size=None) -> draw samples from a Standard Normal distribution
    np.random.standard_t,     # standard_t(df, size=None) -> draw samples from a Standard Student's t distribution
    np.random.triangular,     # triangular(left, mode, right, size=None) -> draw samples from a Triangular distribution
    np.random.weibull         # weibull(a, size=None) -> draw samples from a Weibull distribution
]

# NumPy FFT (Fast Fourier Transform) Functions

numpy_fft_functions = [
    np.fft.fft,                # fft(a, n=None, axis=-1, norm=None) -> compute the one-dimensional discrete Fourier Transform
    np.fft.ifft,               # ifft(a, n=None, axis=-1, norm=None) -> compute the one-dimensional inverse discrete Fourier Transform
    np.fft.fft2,               # fft2(x, s=None, axes=(-2, -1), norm=None) -> compute the two-dimensional discrete Fourier Transform
    np.fft.ifft2,              # ifft2(x, s=None, axes=(-2, -1), norm=None) -> compute the two-dimensional inverse discrete Fourier Transform
    np.fft.fftshift,           # fftshift(x, axes=None) -> shift the zero frequency component to the center of the spectrum
    np.fft.ifftshift,          # ifftshift(x, axes=None) -> shift the zero frequency component back to the original position
    np.fft.fftn,               # fftn(a, s=None, axes=None, norm=None) -> compute the N-dimensional discrete Fourier Transform
    np.fft.ifftn,              # ifftn(a, s=None, axes=None, norm=None) -> compute the N-dimensional inverse discrete Fourier Transform
    np.fft.rfft,               # rfft(x, n=None, axis=-1, norm=None) -> compute the one-dimensional discrete Fourier Transform for real input
    np.fft.irfft,              # irfft(x, n=None, axis=-1, norm=None) -> compute the one-dimensional inverse discrete Fourier Transform for real input
    np.fft.rfftfreq,           # rfftfreq(n, d=1.0) -> return the Discrete Fourier Transform sample frequencies for real input
    np.fft.fftfreq,            # fftfreq(n, d=1.0) -> return the Discrete Fourier Transform sample frequencies
    np.fft.hfft,               # hfft(x, n=None) -> compute the discrete Fourier Transform of a real input signal using Hermitian symmetry
    np.fft.ihfft               # ihfft(x, n=None) -> compute the inverse of the discrete Fourier Transform of a real input signal using Hermitian symmetry
]

from physipy.quantity.quantity import HANDLED_FUNCTIONS
physipy_implemented = [f.__name__ for f in HANDLED_FUNCTIONS]
from physipy.quantity.quantity import implemented_ufuncs




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
    
    
    @property
    def func_dict(self):
        return {
            "python_unary_ops":{
                "package":"python",
                "naryness":"unary",
                "funcs":self.python_unary_ops,
            },
            "python_binary_ops":{
                "package":"python",
                "naryness":"binary",
                "funcs":self.python_binary_ops,
            },
            "numpy_unary_ops":{
                "package":"numpy",
                "naryness":"unary",
                "funcs":self.numpy_unary_ufuncs,
            },
            "numpy_binary_ops":{
                "package":"numpy",
                "naryness":"binary",
                "funcs":self.numpy_binary_ufuncs,
            },
            "numpy_funcs":{
                "package":"numpy",
                "naryness":"unary",
                "funcs":self.numpy_array_function,
            },

        }

    
    def __init__(self, dtype=np.float64):
        self.dtype = dtype
        
    
    @property
    def numerical_values_dict(self):
        return {
            "scalar":self.scalars,
            "array":self.set_arrays(),
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

    