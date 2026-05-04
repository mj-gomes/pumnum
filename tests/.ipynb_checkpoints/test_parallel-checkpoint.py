import sys
from pumnum import pumnum
import pint
import unyt
from numba import njit, prange
import numpy as np

@pumnum(parallel=True)
def function1_pumnum(a, b, c):
    aa=0
    for i in prange(1000):
        aa+=(a * b / c**3 / 2)
    return aa

@njit(parallel=True)
def function1_njit(a, b, c):
    aa=0
    for i in prange(1000):
        aa+=(a * b / c**3 / 2)
    return aa

ureg = pint.UnitRegistry()

a_pint=2.5*ureg.m
b_pint=3*ureg.km
c_pint=10*ureg.s

a_unyt=2.5*unyt.m
b_unyt=3*unyt.km
c_unyt=10*unyt.s

result_pumnum_pint = function1_pumnum(a_pint, b_pint, c_pint)
result_pumnum_unyt = function1_pumnum(a_unyt, b_unyt, c_unyt)
result_njit = function1_njit(a_pint.magnitude, b_pint.magnitude, c_pint.magnitude)

np.testing.assert_equal(result_pumnum_pint.magnitude, result_njit)
np.testing.assert_equal(result_pumnum_unyt, result_njit)
