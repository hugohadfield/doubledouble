from __future__ import print_function
from decimal import Decimal, getcontext
from doubledouble_numba import *
import time


@numba.njit
def horner_double(polynomial, x):
    s = 0.0
    for c in polynomial:
        s = s*x + c
    return s


@numba.njit
def horner_doubledouble(polynomial, x):
    s = DoubleDouble(0.0, 0.0)
    for c in polynomial:
        s = s*x + c
    return float(s)


def horner_decimal(polynomial, x):
    x = Decimal(x)
    s = Decimal(0.0)
    for c in polynomial:
        s = s*x + Decimal(c)
    return float(s)

getcontext().prec = 100

polynomial, x = (1.0, -78.0, 2717.0, -55770.0, 749463.0, -6926634.0,
    44990231.0, -206070150.0, 657206836.0, -1414014888.0, 1931559552.0,
    -1486442880.0, 479001600.0), 12.001 # Wilkinson's polynomial


for f in (horner_double, horner_doubledouble, horner_decimal):
    print(repr(f(polynomial, x)), '\t', f.__name__)

nreps = 100000
for f in (horner_double, horner_doubledouble, horner_decimal):
    time_start = time.time()
    for i in range(nreps):
        repr(f(polynomial, x))
    time_end = time.time()
    print((time_end-time_start)/nreps)
