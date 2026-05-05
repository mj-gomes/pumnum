import functools
from contextlib import contextmanager

import pint
import unyt
import astropy.units
from numba import njit
import inspect
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

# gets the set of njit arguments, so that the decorator
# can separate between those to pass to numba.njit and
# the ones that are pumnum-specific.
_NJIT_KEYS = set(inspect.signature(njit).parameters.keys())

def pumnum(_func=None, **kwargs):
    njit_kwargs = {k: v for k, v in kwargs.items() if k in _NJIT_KEYS}
    pumnum_kwargs = {k: v for k, v in kwargs.items() if k not in _NJIT_KEYS}

    # set pumnum kwargs
    convert_to = pumnum_kwargs.get("convert_to", None)

    def decorator(func):
        @functools.wraps(func)
        def __init__(*args):
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
                        "it should be either a pint.Quantity, unyt.array.unyt_quantity, or astropy.units.Quantity"
                    )
                args_magnitudes.append(backend(arg).magnitude)
                args_units.append(1 * backend(arg).units)

            loop_once = run_loop_once(func)
            final_units = backend(loop_once(*args_units)).units
            op = njit(**njit_kwargs)(func)
            res = op(*args_magnitudes) * final_units
            if convert_to is not None:
                return backend(res).convert_to_unit_system(convert_to)
            else:
                return res

        return __init__

    if _func is None:
        return decorator
    else:
        return decorator(_func)
