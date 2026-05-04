import numba
import inspect
import ast
from sympy import symbols
import unyt
import functools
from contextlib import contextmanager
from numba import njit

"""
The following functions limit loops to a single iteration.
This is useful for the first pass before the numba jit
compilation, where the final units of the method are
pre-computed
"""

_loop_once = False

@contextmanager
def limit_loops():

    global _loop_once
    old_value = _loop_once
    _loop_once = True
    try:
        yield
    finally:
        _loop_once = old_value

def run_loop_once(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        with limit_loops():
            return func(*args, **kwargs)
    return wrapper

""""""

"""
Definition of the Numyt decorator, that will first get the
final units of the numba jitted function, and then compute it
and give it the pre-computed units
"""

def pumba(func):
    def __init__(*args, **kwargs):
        quantities = []
        for arg in args:
            quantities.append(1*arg.units)

        tmp = run_loop_once(func)
        result = tmp(*quantities)
        units = result.units
        op=numba.njit(func)
        return op(*args)* units
    return __init__
