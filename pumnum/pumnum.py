import functools
from contextlib import contextmanager

import pint
import unyt
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


def pumnum(_func=None, **njit_kwargs):
    def decorator(func):
        @functools.wraps(func)
        def __init__(*args, **kwargs):
            args_magnitudes = []
            args_units = []
            for arg in args:
                if isinstance(arg, pint.Quantity):
                    args_magnitudes.append(arg.magnitude)
                elif isinstance(arg, unyt.array.unyt_quantity):
                    args_magnitudes.append(arg)
                else:
                    return
                args_units.append(1 * arg.units)

            tmp = run_loop_once(func)
            result = tmp(*args_units)
            final_units = result.units
            op = njit(**njit_kwargs)(func)
            return op(*args_magnitudes) * final_units

        return __init__

    if _func is None:
        return decorator
    else:
        return decorator(_func)
