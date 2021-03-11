
import operator

from numba import types
from numba.extending import typeof_impl
from numba.extending import type_callable
from numba.extending import models, register_model
from numba.extending import make_attribute_wrapper
from numba.extending import lower_builtin
from numba.core import cgutils
from numba.extending import unbox, NativeValue
from numba.extending import box

import numpy as np

from doubledouble import *


# Define the numba type
class DoubleDoubleType(types.Type):
    def __init__(self):
        super(DoubleDoubleType, self).__init__(name='DoubleDouble')

doubledouble_type = DoubleDoubleType()



@typeof_impl.register(DoubleDouble)
def typeof_index(val, c):
    return doubledouble_type


@type_callable(DoubleDouble)
def type_doubledouble(context):
    def typer(x, y):
        if isinstance(x, types.Float) and isinstance(y, types.Float):
            return doubledouble_type
    return typer


@register_model(DoubleDoubleType)
class DoubleDoubleModel(models.StructModel):
    def __init__(self, dmm, fe_type):
        members = [
            ('x', types.float64),
            ('y', types.float64),
            ]
        models.StructModel.__init__(self, dmm, fe_type, members)


make_attribute_wrapper(DoubleDoubleType, 'x', 'x')
make_attribute_wrapper(DoubleDoubleType, 'y', 'y')


@lower_builtin(DoubleDouble, types.Float, types.Float)
def impl_doubledouble(context, builder, sig, args):
    typ = sig.return_type
    x, y = args
    doubledouble = cgutils.create_struct_proxy(typ)(context, builder)
    doubledouble.x = x
    doubledouble.y = y
    return doubledouble._getvalue()


@unbox(DoubleDoubleType)
def unbox_doubledouble(typ, obj, c):
    """
    Convert a DoubleDouble object to a native doubledouble structure.
    """
    x_obj = c.pyapi.object_getattr_string(obj, "x")
    y_obj = c.pyapi.object_getattr_string(obj, "y")
    doubledouble = cgutils.create_struct_proxy(typ)(c.context, c.builder)
    doubledouble.x = c.pyapi.float_as_double(x_obj)
    doubledouble.y = c.pyapi.float_as_double(y_obj)
    c.pyapi.decref(x_obj)
    c.pyapi.decref(y_obj)
    is_error = cgutils.is_not_null(c.builder, c.pyapi.err_occurred())
    return NativeValue(doubledouble._getvalue(), is_error=is_error)


@box(DoubleDoubleType)
def box_doubledouble(typ, val, c):
    """
    Convert a native doubledouble structure to an DoubleDouble object.
    """
    doubledouble = cgutils.create_struct_proxy(typ)(c.context, c.builder, value=val)
    x_obj = c.pyapi.float_from_double(doubledouble.x)
    y_obj = c.pyapi.float_from_double(doubledouble.y)
    class_obj = c.pyapi.unserialize(c.pyapi.serialize_object(DoubleDouble))
    res = c.pyapi.call_function_objargs(class_obj, (x_obj, y_obj))
    c.pyapi.decref(x_obj)
    c.pyapi.decref(y_obj)
    c.pyapi.decref(class_obj)
    return res


@numba.extending.overload(operator.mul)
def double_double_mul(a, b):
    if isinstance(a, DoubleDoubleType) and isinstance(b, DoubleDoubleType):
        def impl(a, b):
            r, e = mul_double_double(a.x, b.x, a.y, b.y)
            return DoubleDouble(r, e)
        return impl
    elif isinstance(a, types.abstract.Number) and isinstance(b, DoubleDoubleType):
        def impl(a, b):
            r, e = rmul_double_double(b.x, b.y, a)
            return DoubleDouble(r, e)
        return impl
    elif isinstance(a, DoubleDoubleType) and isinstance(b, types.abstract.Number):
        def impl(a, b):
            r, e = rmul_double_double(a.x, a.y, b)
            return DoubleDouble(r, e)
        return impl


@numba.extending.overload(operator.add)
def double_double_add(a, b):
    if isinstance(a, DoubleDoubleType) and isinstance(b, DoubleDoubleType):
        def impl(a, b):
            r, e = add_double_double(a.x, b.x, a.y, b.y)
            return DoubleDouble(r, e)
        return impl
    elif isinstance(a, types.abstract.Number) and isinstance(b, DoubleDoubleType):
        def impl(a, b):
            r, e = radd_double_double(b.x, b.y, a)
            return DoubleDouble(r, e)
        return impl
    elif isinstance(a, DoubleDoubleType) and isinstance(b, types.abstract.Number):
        def impl(a, b):
            r, e = radd_double_double(a.x, a.y, b)
            return DoubleDouble(r, e)
        return impl


@numba.extending.overload(float)
def doubledouble_float(a):
    if isinstance(a, DoubleDoubleType):
        def impl(a):
            return a.x
        return impl


if __name__ == '__main__':
    @numba.njit
    def test(a, b):
        return 2.0 + a*b + 1.0

    print(test(DoubleDouble(1.0), DoubleDouble(2.0)))
