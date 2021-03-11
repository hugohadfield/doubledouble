
import numpy as np
from doubledouble import mul_double_double
import numba
import numba.np.numpy_support as _numpy_support
from numba import types
import operator


np_type = np.dtype([('x', np.float64), ('y', np.float64)])
zero = np.zeros(1, dtype=np_type)[0]
numba_dtype = numba.from_dtype(np_type)
numba_array_type = numba.from_dtype(np_type)[:]


# print(numba_dtype)
# print(dir(numba_dtype))
# print(numba_dtype.dtype)
# print(numba_dtype())
# exit()
array_type = type(zero)


@numba.njit
def numpy_mul_double_double(a, b):
    r, e = mul_double_double(a['x'], b['x'], a['y'], b['y'])
    out = np.zeros(1, dtype=np_type)[0]
    out['x'] = r
    out['y'] = e
    return out


@numba.extending.overload(operator.mul)
def np_double_double_mul(a, b):
    if isinstance(a, numba.core.types.npytypes.Record) and isinstance(b, numba.core.types.npytypes.Record):
        def impl(a, b):
            return numpy_mul_double_double(a, b)
        return impl


@numba.njit
def alt_test(a, b):
    return a*b

#
np_elem = np.zeros(1, dtype=np_type)[0]
np_elem['x'] = np.random.randn()
np_elem['y'] = np.random.randn()
# b = np.zeros(1, dtype=np_type)[0]
#
#
# b['x'] = np.random.randn()
# a['y'] = np.random.randn()
# b['y'] = np.random.randn()
# print(a)
# print(b)
#
# print(numpy_mul_double_double(a, b))
#
# c = np.array([zero, zero])
# d = np.array([zero, zero])
# print(alt_test(a, d))

from clifford.g3c import *

a = 1.0*e1
a.value = np.array([np_elem for i in range(32)])

print(a.value)

b = 1.0*e1
b.value = np.array([np_elem for i in range(32)])

print(b.value)

print((a*b).value)