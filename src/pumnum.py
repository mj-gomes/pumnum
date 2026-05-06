import functools
from contextlib import contextmanager
import ast
import inspect
import sys

import pint
import unyt
import astropy.units
from numba import njit

from backends import BkAstropy, BkPint, BkUnyt


def is_nopython_compatible(x):
    @njit
    def _test_njit_compat(x):
        # if we don't do an operation, njit will return the
        # input object even if it is not compatible
        x=x+0
        return x

    try:
        y = _test_njit_compat(x)
        # for some objects, njit takes their units part,
        # even if it does not return an Exception.
        # so we have to check if the quantity is the same
        # after passing through a njit function.
        return x==y
    except Exception:
        return False

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

registry = {"precompute_units": False}
_NJIT_KEYS = set(inspect.signature(njit).parameters.keys())

def _collect_calls(func):
    """
    This function uses inspect to get the input function
    written as a string, and then the ast module will parse
    it and walk through it in order to find instances of
    function calls (ast.Call).

    Returns a set of function id's (i.e. a str with the name of
    the function)
    """
    try:
        src = inspect.getsource(func)
    except (OSError, IOError, TypeError, OSError):
        return None
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return None

    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            # case where it is a simple function call
            # e.g. function(). in that case it is a
            # ast.Name object.
            if isinstance(fn, ast.Name):
                names.add(fn.id)
            # case where it is a function called has an
            # attribute of an object, e.g. obj.function()
            # in this case we have to go part by part.
            elif isinstance(fn, ast.Attribute):
                parts = []
                cur = fn
                while isinstance(cur, ast.Attribute):
                    parts.append(cur.attr)
                    cur = cur.value
                if isinstance(cur, ast.Name):
                    parts.append(cur.id)
                    parts = list(reversed(parts))
                    names.add(("attr", tuple(parts)))

    return names

def _resolve_attribute_candidate(mod, parts):
    """
    This function resolves the multiple parts of calls to object/module methods.
    Returns the final object or None.
    """
    obj = None
    first = parts[0]
    if not hasattr(mod, first):
        return None
    obj = getattr(mod, first)
    for p in parts[1:]:
        if not hasattr(obj, p):
            return None
        obj = getattr(obj, p)
    return obj



def _replace_pumnum_with_njit(func, njit_kwargs):
    """
    This function inspects the pumnum decorated function (func) to find calls to
    other pumnum decorated functions, and changes those to @njit decorated functions.

    This happens when we want to compute func with njit, after the first pass
    of the pumnum decorator that obtains the final units. We don't want the functions
    that are called inside func to be @pumnum decorated at this step, because they would
    return quantities with physical units that would result in a numba nopython error.

    This function works in three major steps:
        1 - use a ast walk to fetch function calls inside func (_collect_calls)
        2 - check which of those functions are @pumnum decorated (using is_pumnum_decorated attribute)
        3 - changing those functions to @njit decorated, instead of @pumnum

    Important: this function returns a dict with the previous @pumnum functions,
    that will be used after func computation ends, to return them back to the original state.
    """

    replaced = {}
    module = sys.modules.get(func.__module__)
    if module is None:
        return replaced

    names = _collect_calls(func)

    for entry in names:
        # First case: "simple" function call, not a method from some object/module
        if not(isinstance(entry, tuple) and entry and entry[0] == "attr"):
            name = entry
            if not isinstance(name, str):
                continue
            candidate = getattr(module, name, None)
            if candidate is None:
                continue
            if not hasattr(candidate, "is_pumnum_decorated"):
                continue
            orig_callee = getattr(candidate, "__wrapped__", candidate)
            op = getattr(candidate, "_pumnum_njit_op", None)
            if op is None:
                op = njit(**njit_kwargs)(orig_callee)
                setattr(candidate, "_pumnum_njit_op", op)
            # replace and save original to restore later
            saved = getattr(module, name, None)
            if saved is not op:
                setattr(module, name, op)
                replaced[name] = saved

        else:
            # Case where the function is a method from some object/module
            parts = entry[1]
            target = _resolve_attribute_candidate(module, parts)
            if target is None:
                continue
            top_name = parts[0]
            callee_name = parts[-1]
            parent_obj = getattr(module, top_name, None)
            if parent_obj is None:
                continue
            candidate = target
            if not hasattr(candidate, "is_pumnum_decorated"):
                continue
            orig_callee = getattr(candidate, "__wrapped__", candidate)
            op = getattr(candidate, "_pumnum_njit_op", None)
            if op is None:
                op = njit(**njit_kwargs)(orig_callee)
                setattr(candidate, "_pumnum_njit_op", op)
            try:
                saved = getattr(parent_obj, callee_name, None)
                if saved is not op:
                    setattr(parent_obj, callee_name, op)
                    replaced[(top_name, callee_name)] = saved
            except Exception:
                # TODO
                continue


    return replaced

def _restore_replaced_functions(orig_func_module, replaced):
    if not replaced:
        return
    module = sys.modules.get(orig_func_module)
    if module is None:
        return
    for k, v in replaced.items():
        if isinstance(k, tuple):
            top_name, attr_name = k
            parent = getattr(module, top_name, None)
            if parent is not None:
                try:
                    # restore attribute
                    setattr(parent, attr_name, v)
                except Exception:
                    pass
        else:
            try:
                setattr(module, k, v)
            except Exception:
                pass

def pumnum(_func=None, **kwargs):
    njit_kwargs = {k: v for k, v in kwargs.items() if k in _NJIT_KEYS}
    pumnum_kwargs = {k: v for k, v in kwargs.items() if k not in _NJIT_KEYS}
    convert_to = pumnum_kwargs.get("convert_to", None)

    def decorator(func):
        setattr(func, "is_pumnum_decorated", True)
        @functools.wraps(func)
        def wrapper(*args, **call_kwargs):
            orig = getattr(func, "__wrapped__", func)

            # detect unit-bearing args and prepare magnitudes + backend
            has_units = False
            args_magnitudes = []
            args_units = []
            backend = None
            for arg in args:
                if is_nopython_compatible(arg):
                    args_magnitudes.append(arg)
                elif isinstance(arg, pint.Quantity):
                    has_units = True
                    backend = BkPint
                    args_magnitudes.append(backend(arg).magnitude)
                    args_units.append(1 * backend(arg).units)
                elif isinstance(arg, unyt.array.unyt_quantity):
                    has_units = True
                    backend = BkUnyt
                    args_magnitudes.append(backend(arg).magnitude)
                    args_units.append(1 * backend(arg).units)
                elif isinstance(arg, astropy.units.Quantity):
                    has_units = True
                    backend = BkAstropy
                    args_magnitudes.append(backend(arg).magnitude)
                    args_units.append(1 * backend(arg).units)
                else:
                    raise TypeError(
                            f"object of type {type(arg).__name__} is not compatible with pumnum; "
                            "it should be either a numba njit compatible object or one of type"
                            " pint.Quantity, unyt.array.unyt_quantity, or astropy.units.Quantity."
                        )


            if registry["precompute_units"]:
                if has_units:
                    orig = getattr(func, "__wrapped__", func)
                    return orig(*args_units, **call_kwargs)
                else:
                    return None

            final_unit = None
            if has_units:
                old = registry["precompute_units"]
                registry["precompute_units"] = True
                try:
                    orig = getattr(func, "__wrapped__", func)
                    final = orig(*args_units, **call_kwargs)
                finally:
                    registry["precompute_units"] = old

                final_unit = backend(final).units

            replaced = _replace_pumnum_with_njit(orig, njit_kwargs)
            try:
                op = getattr(func, "_pumnum_njit_op", None)
                if op is None:
                    op = njit(**njit_kwargs)(orig)
                    setattr(func, "_pumnum_njit_op", op)

                # call compiled op on magnitudes
                result_magnitude = op(*args_magnitudes)
            finally:
                _restore_replaced_functions(orig.__module__, replaced)

            # attach precomputed units
            if final_unit is not None:
                res = result_magnitude * final_unit
                if convert_to is not None:
                    return backend(res).convert_to_unit_system(convert_to)
                else:
                    return res
            else:
                return result_magnitude

        return wrapper

    if _func is None:
        return decorator
    else:
        return decorator(_func)
