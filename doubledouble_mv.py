
import numpy as np
import numba
import clifford as cf
from doubledouble import DoubleDouble

def get_shirokov_inverse(layout):
    n = len(layout.sig)
    exponent = (n + 1) // 2
    N = 2 ** exponent
    def shirokov_inverse(U):
        Uk = U * 1.0  # cast to float
        for k in range(1, N):
            Ck = (N / k) * Uk.value[0]
            adjU = (Uk - Ck.x)
            Uk = U * adjU
        if Uk.value[0] == 0:
            raise ValueError('Multivector has no inverse')
        return adjU / Uk.value[0]
    return shirokov_inverse


def get_shirokov_inverse_dd(layout):
    n = len(layout.sig)
    exponent = (n + 1) // 2
    N = 2 ** exponent
    Ndd = DoubleDouble(1.0 * N)
    def shirokov_inverse(U):
        Uk = U * 1.0  # cast to float
        for k in range(1, N):
            Ck = (Ndd / DoubleDouble(1.0 * k)) * DoubleDouble(Uk.x.value[0], Uk.y.value[0])
            adjU = Uk - Ck
            Uk = U * adjU
        if DoubleDouble(Uk.x.value[0], Uk.y.value[0]) == 0:
            raise ValueError('Multivector has no inverse')
        return adjU / DoubleDouble(Uk.x.value[0], Uk.y.value[0])
    return shirokov_inverse


rng = np.random.default_rng(1)

layout, blades = cf.Cl(8, 0, 1)

sinv = get_shirokov_inverse(layout)
sinv_dd = get_shirokov_inverse_dd(layout)

for i in range(10):
    avalue = np.array([DoubleDouble(x) for x in rng.standard_normal(layout.gaDims)], dtype=object)
    a = layout.MultiVector(avalue)
    ainv = sinv(a)
    print(np.linalg.norm((1 - a*ainv).value))

    # add = DoubleDouble(a)
    # ainvdd = sinv_dd(add)
    # res_dd = (1 - add*ainvdd)
    # print(np.linalg.norm((1 - a*ainv).value), np.linalg.norm(res_dd.x.value))