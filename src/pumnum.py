import functools
from contextlib import contextmanager

import pint
import unyt
import astropy.units
from numba import njit

from backends import BkAstropy, BkPint, BkUnyt

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
Definition of the pumnum decorator, that will first get the
final units of the numba jitted function, and then compute it
and give it the pre-computed units
"""


def pumnum(_func=None, **njit_kwargs):
    def decorator(func):
        @functools.wraps(func)
        def __init__(*args, **kwargs):
            args_magnitudes = []
            args_units = []
            for idx, arg in enumerate(args):
                if isinstance(arg, pint.Quantity):
                    backend=BkPint
                elif isinstance(arg, unyt.array.unyt_quantity):
                    backend=BkUnyt
                elif isinstance(arg, astropy.units.Quantity):
                    backend=BkAstropy
                else:
                    raise TypeError(
                        f"object of type {type(obj).__name__} is not compatible with pumnum; "
                        "it should be either a pint.Quantity, unyt.array, or astropy.units.Quantity"
                    )
                args_magnitudes.append(backend(arg).value)
                args_units.append(1 * backend(arg).units)

            tmp = run_loop_once(func)
            result = tmp(*args_units)
            final_units = backend(result).units
            op = njit(**njit_kwargs)(func)
            return op(*args_magnitudes) * final_units

        return __init__

    if _func is None:
        return decorator
    else:
        return decorator(_func)
