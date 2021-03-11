
import numpy as np
import numba
import clifford as cf
from clifford import mul_double_double, truediv_doubledouble, sub_double_double
from doubledouble import DoubleDouble


def get_shirokov_inverse(layout):
    n = len(layout.sig)
    exponent = (n + 1) // 2
    N = 2 ** exponent

    gmt_dd = layout.gmt_func_double_double()
    @numba.njit
    def shirokov_inverse(U):
        Uval = np.zeros((U.value.shape[0], 2), U.value.dtype)
        Uval[:, 0] = U.value * 1.0
        Ukval = np.zeros((U.value.shape[0], 2), U.value.dtype)
        Ukval[:, 0] = U.value * 1.0
        for k in range(1, N):
            Nkr, Nke = truediv_doubledouble(1.0*N, 1.0*k, 0.0, 0.0)
            Ckr, Cke = mul_double_double(Ukval[0, 0], Nkr, Ukval[0, 1], Nke)
            adjU = Ukval
            r, e = sub_double_double(adjU[0, 0], Ckr, adjU[0, 1], Cke)
            adjU[0, 0] = r
            adjU[0, 1] = e
            Ukval = gmt_dd(Uval, adjU)
        if np.all(Ukval[0, :] == 0):
            raise ValueError('Multivector has no inverse')
        denomr, denome = Ukval[0, :]
        for i in range(adjU.shape[0]):
            r, e = truediv_doubledouble(adjU[i, 0], denomr, adjU[i, 1], denome)
            adjU[i, 0] = r
            adjU[i, 1] = e
        return layout.MultiVector(adjU[:, 0])
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
    avalue = rng.standard_normal(layout.gaDims)
    a = layout.MultiVector(avalue)
    ainv = sinv(a)
    print(np.linalg.norm((1 - a * ainv).value))

#
# for i in range(10):
#     avalue = np.array([DoubleDouble(x) for x in rng.standard_normal(layout.gaDims)], dtype=object)
#     a = layout.MultiVector(avalue)
#     ainv = sinv(a)
#     print(np.linalg.norm((1 - a*ainv).value))

    # add = DoubleDouble(a)
    # ainvdd = sinv_dd(add)
    # res_dd = (1 - add*ainvdd)
    # print(np.linalg.norm((1 - a*ainv).value), np.linalg.norm(res_dd.x.value))