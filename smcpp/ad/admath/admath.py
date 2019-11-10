# -*- coding: utf-8 -*-
"""
Mathematical operations that generalize many operations from the standard math
and cmath modules so that they also track first and second derivatives.

The basic philosophy of order of type-operations is this:
A. Is X from the ADF class or subclass? 
   1. Yes - Perform automatic differentiation.
   2. No - Is X an array object?
      a. Yes - Vectorize the operation and repeat at A for each item.
      b. No - Let the math/cmath function deal with X since it's probably a base
         numeric type. Otherwise they will throw the respective exceptions.

Examples:

  from admath import sin
  
  # Manipulation of numbers that track derivatives:
  x = ad.adnumber(3)
  print sin(x)  # prints ad(0.1411200080598672)

  # The admath functions also work on regular Python numeric types:
  print sin(3)  # prints 0.1411200080598672.  This is a normal Python float.

Importing all the functions from this module into the global namespace
is possible.  This is encouraged when using a Python shell as a
calculator.  Example:

  import ad
  from ad.admath import *  # Imports tan(), etc.
  
  x = ad.adnumber(3)
  print tan(x)  # tan() is the ad.admath.tan function

The numbers with derivative tracking handled by this module are objects from
the ad (automatic differentiation) module, from either the ADV or the ADF class.

(c) 2013 by Abraham Lee <tisimst@gmail.com>.
Please send feature requests, bug reports, or feedback to this address.

This software is released under a dual license.  (1) The BSD license.
(2) Any other license, as long as it is obtained from the original
author.

"""
from __future__ import division
import math
import cmath
from .. import __author__, ADF, to_auto_diff, _apply_chain_rule

try:
    import numpy as np
except ImportError:
    numpy_installed = False
else:
    numpy_installed = True
    
    def return_numpy_array_if_given(func):
        def wrapped_func(*args):
            if len(args)==1:
                ans = func(*args)
                if isinstance(args[0], np.ndarray):
                    ans = np.array(ans)
            else:
                same_arg_lengths = True
                for arg1 in args[:-1]:
                    for arg2 in args[1:]:
                        if len(arg1)!=len(arg2):
                            same_arg_lengths = False
                            break
                if same_arg_lengths:
                    ans = [func(*arg) for arg in args]
                    if any([isinstance(arg, np.ndarray) for arg in args]):
                        ans = np.array(ans)
                else:
                    raise ValueError('Input arguments not the same length')
            return ans
        
        wrapped_func.__name__ = func.__name__
        wrapped_func.__module__ = func.__module__
        if func.__doc__ is not None and isinstance(func.__doc__, str):
            wrapped_func.__doc__ = func.__doc__
        
        return wrapped_func
                            
                            
__all__ = [
    # math/cmath module equivalent functions
    'sin', 'asin', 'sinh', 'asinh',
    'cos', 'acos', 'cosh', 'acosh',
    'tan', 'atan', 'atan2', 'tanh', 'atanh',
    'e', 'pi', 
    'isinf', 'isnan', 
    'phase', 'polar', 'rect',
    'exp', 'expm1',
    'erf', 'erfc',
    'factorial', 'gamma', 'lgamma',
    'log', 'ln', 'log10', 'log1p',
    'sqrt', 'hypot', 'pow',
    'degrees', 'radians',
    'ceil', 'floor', 'trunc', 'fabs',
    # other miscellaneous functions that are conveniently defined
    'csc', 'acsc', 'csch', 'acsch',
    'sec', 'asec', 'sech', 'asech',
    'cot', 'acot', 'coth', 'acoth'
    ]

            
### FUNCTIONS IN THE MATH MODULE ##############################################
#
# Currently, there is no implementation for the following math module methods:
# - copysign
# - factorial <- depends on gamma
# - fmod
# - frexp
# - fsum
# - gamma* <- currently uses high-accuracy finite difference derivatives
# - lgamma <- depends on gamma
# - ldexp
# - modf
#
# we'll see if they(*) get implemented

e = math.e
pi = math.pi

# for convenience, (though totally defeating the purpose of having an 
# automatic differentiation class, define a fourth-order finite difference 
# derivative for when an analytical derivative doesn't exist
eps = 1e-8 # arbitrarily chosen
def _fourth_order_first_fd(func,x):
    fm2 = func(x-2*eps)
    fm1 = func(x-eps)
    fp1 = func(x+eps)
    fp2 = func(x+2*eps)
    return (fm2-8*fm1+8*fp1-fp2)/12/eps

def _fourth_order_second_fd(func,x):
    fm2 = func(x-2*eps)
    fm1 = func(x-eps)
    f   = func(x)
    fp1 = func(x+eps)
    fp2 = func(x+2*eps)
    return (-fm2+16*fm1-30*f+16*fp1-fp2)/12/eps**2
    
def _vectorize(func):
    """
    Make a function that accepts 1 or 2 arguments work with input arrays (of
    length m) in the following array length combinations:
    
    - m x m
    - 1 x m
    - m x 1
    - 1 x 1
    """
    def vectorized_function(*args, **kwargs):
        if len(args)==1:
            x = args[0]
            try:
                return [vectorized_function(xi, **kwargs) for xi in x]
            except TypeError:
                return func(x, **kwargs)
                
        elif len(args)==2:
            x, y = args
            try:
                return [vectorized_function(xi, yi, **kwargs) 
                        for xi, yi in zip(x, y)]
            except TypeError:
                try:
                    return [vectorized_function(xi, y, **kwargs) for xi in x]
                except TypeError:
                    try:
                        return [vectorized_function(x, yi, **kwargs) for yi in y]
                    except TypeError:
                        return func(x, y, **kwargs)
    
    n = func.__name__
    m = func.__module__
    d = func.__doc__
    
    vectorized_function.__name__ = n
    vectorized_function.__module__ = m
    doc = 'Vectorized {0:} function\n'.format(n)
    if d is not None:
        doc += d
    vectorized_function.__doc__ = doc
            
    return vectorized_function
            
@_vectorize
def acos(x):
    """
    Return the arc cosine of x, in radians.
    """
    if isinstance(x,ADF):
        ad_funcs = list(map(to_auto_diff,[x]))

        x = ad_funcs[0].x
        
        ########################################
        # Nominal value of the constructed ADF:
        f = acos(x)
        
        ########################################

        variables = ad_funcs[0]._get_variables(ad_funcs)
        
        if not variables or isinstance(f, bool):
            return f

        ########################################

        # Calculation of the derivatives with respect to the arguments
        # of f (ad_funcs):

        lc_wrt_args = [-1/sqrt(1-x**2)]
        qc_wrt_args = [x/(sqrt(1 - x**2)*(x**2 - 1))]
        cp_wrt_args = 0.0

        ########################################
        # Calculation of the derivative of f with respect to all the
        # variables (Variable) involved.


        lc_wrt_vars,qc_wrt_vars,cp_wrt_vars = _apply_chain_rule(
                                    ad_funcs,variables,lc_wrt_args,qc_wrt_args,
                                    cp_wrt_args)
                                    
        # The function now returns an ADF object:
        return ADF(f, lc_wrt_vars, qc_wrt_vars, cp_wrt_vars)
    else:
#        try: # pythonic: fails gracefully when x is not an array-like object
#            return [acos(xi) for xi in x]
#        except TypeError:
        if x.imag:
            return cmath.acos(x)
        else:
            return math.acos(x.real)

@_vectorize
def acosh(x):
    """
    Return the inverse hyperbolic cosine of x.
    """
    if isinstance(x,ADF):
        ad_funcs = list(map(to_auto_diff,[x]))

        x = ad_funcs[0].x
        
        ########################################
        # Nominal value of the constructed ADF:
        f = acosh(x)
        
        ########################################

        variables = ad_funcs[0]._get_variables(ad_funcs)
        
        if not variables or isinstance(f, bool):
            return f

        ########################################

        # Calculation of the derivatives with respect to the arguments
        # of f (ad_funcs):

        lc_wrt_args = [1/sqrt(x**2 - 1)]
        qc_wrt_args = [-x/(x**2 - 1)**1.5]
        cp_wrt_args = 0.0

        ########################################
        # Calculation of the derivative of f with respect to all the
        # variables (Variable) involved.

        lc_wrt_vars,qc_wrt_vars,cp_wrt_vars = _apply_chain_rule(
                                    ad_funcs,variables,lc_wrt_args,qc_wrt_args,
                                    cp_wrt_args)
                                    
        # The function now returns an ADF object:
        return ADF(f, lc_wrt_vars, qc_wrt_vars, cp_wrt_vars)
    else:
#        try: # pythonic: fails gracefully when x is not an array-like object
#            return [acosh(xi) for xi in x]
#        except TypeError:
        if x.imag:
            return cmath.acosh(x)
        else:
            return math.acosh(x.real)

@_vectorize
def asin(x):
    """
    Return the arc sine of x, in radians.
    """
    if isinstance(x,ADF):
        ad_funcs = list(map(to_auto_diff,[x]))

        x = ad_funcs[0].x
        
        ########################################
        # Nominal value of the constructed ADF:
        f = asin(x)
        
        ########################################

        variables = ad_funcs[0]._get_variables(ad_funcs)
        
        if not variables or isinstance(f, bool):
            return f

        ########################################

        # Calculation of the derivatives with respect to the arguments
        # of f (ad_funcs):

        lc_wrt_args = [1/sqrt(1 - x**2)]
        qc_wrt_args = [-x/(sqrt(1 - x**2)*(x**2 - 1))]
        cp_wrt_args = 0.0

        ########################################
        # Calculation of the derivative of f with respect to all the
        # variables (Variable) involved.

        lc_wrt_vars,qc_wrt_vars,cp_wrt_vars = _apply_chain_rule(
                                    ad_funcs,variables,lc_wrt_args,qc_wrt_args,
                                    cp_wrt_args)
                                    
        # The function now returns an ADF object:
        return ADF(f, lc_wrt_vars, qc_wrt_vars, cp_wrt_vars)
    else:
#        try: # pythonic: fails gracefully when x is not an array-like object
#            return [asin(xi) for xi in x]
#        except TypeError:
        if x.imag:
            return cmath.asin(x)
        else:
            return math.asin(x.real)

@_vectorize
def asinh(x):
    """
    Return the inverse hyperbolic sine of x.
    """
    if isinstance(x,ADF):
        ad_funcs = list(map(to_auto_diff,[x]))

        x = ad_funcs[0].x
        
        ########################################
        # Nominal value of the constructed ADF:
        f = asinh(x)
        
        ########################################

        variables = ad_funcs[0]._get_variables(ad_funcs)
        
        if not variables or isinstance(f, bool):
            return f

        ########################################

        # Calculation of the derivatives with respect to the arguments
        # of f (ad_funcs):

        lc_wrt_args = [1/sqrt(x**2 + 1)]
        qc_wrt_args = [-x/(x**2 + 1)**1.5]
        cp_wrt_args = 0.0

        ########################################
        # Calculation of the derivative of f with respect to all the
        # variables (Variable) involved.

        lc_wrt_vars,qc_wrt_vars,cp_wrt_vars = _apply_chain_rule(
                                    ad_funcs,variables,lc_wrt_args,qc_wrt_args,
                                    cp_wrt_args)
                                    
        # The function now returns an ADF object:
        return ADF(f, lc_wrt_vars, qc_wrt_vars, cp_wrt_vars)
    else:
#        try: # pythonic: fails gracefully when x is not an array-like object
#            return [asinh(xi) for xi in x]
#        except TypeError:
        if x.imag:
            return cmath.asinh(x)
        else:
            return math.asinh(x.real)

@_vectorize
def atan(x):
    """
    Return the arc tangent of x, in radians
    """
    if isinstance(x,ADF):
        ad_funcs = list(map(to_auto_diff,[x]))

        x = ad_funcs[0].x
        
        ########################################
        # Nominal value of the constructed ADF:
        f = atan(x)
        
        ########################################

        variables = ad_funcs[0]._get_variables(ad_funcs)
        
        if not variables or isinstance(f, bool):
            return f

        ########################################

        # Calculation of the derivatives with respect to the arguments
        # of f (ad_funcs):

        lc_wrt_args = [1/(x**2 + 1)]
        qc_wrt_args = [-2*x/(x**4 + 2*x**2 + 1)]
        cp_wrt_args = 0.0

        ########################################
        # Calculation of the derivative of f with respect to all the
        # variables (Variable) involved.

        lc_wrt_vars,qc_wrt_vars,cp_wrt_vars = _apply_chain_rule(
                                    ad_funcs,variables,lc_wrt_args,qc_wrt_args,
                                    cp_wrt_args)
                                    
        # The function now returns an ADF object:
        return ADF(f, lc_wrt_vars, qc_wrt_vars, cp_wrt_vars)
    else:
#        try: # pythonic: fails gracefully when x is not an array-like object
#            return [atan(xi) for xi in x]
#        except TypeError:
        if x.imag:
            return cmath.atan(x)
        else:
            return math.atan(x.real)

@_vectorize
def atan2(y, x):
    """
    Return ``atan(y / x)``, in radians. The result is between ``-pi`` and 
    ``pi``. The vector in the plane from the origin to point ``(x, y)`` makes 
    this angle with the positive X axis. The point of ``atan2()`` is that the 
    signs of both inputs are known to it, so it can compute the correct 
    quadrant for the angle. For example, ``atan(1)`` and ``atan2(1, 1)`` are 
    both ``pi/4``, but ``atan2(-1, -1)`` is ``-3*pi/4``.
    """
    if x>0:
        return atan(y/x)
    elif x<0:
        if y>=0:
            return atan(y/x) + pi
        else:
            return atan(y/x) - pi
    else:
        if y>0:
            return +pi/2
        elif y<0:
            return -pi/2
        else:
            return 0.0
    
@_vectorize
def atanh(x):
    """
    Return the inverse hyperbolic tangent of x.
    """
    if isinstance(x,ADF):
        ad_funcs = list(map(to_auto_diff,[x]))

        x = ad_funcs[0].x
        
        ########################################
        # Nominal value of the constructed ADF:
        f = atanh(x)
        
        ########################################

        variables = ad_funcs[0]._get_variables(ad_funcs)
        
        if not variables or isinstance(f, bool):
            return f

        ########################################

        # Calculation of the derivatives with respect to the arguments
        # of f (ad_funcs):

        lc_wrt_args = [-1./(x**2 - 1)]
        qc_wrt_args = [2*x/(x**4 - 2*x**2 + 1)]
        cp_wrt_args = 0.0

        ########################################
        # Calculation of the derivative of f with respect to all the
        # variables (Variable) involved.

        lc_wrt_vars,qc_wrt_vars,cp_wrt_vars = _apply_chain_rule(
                                    ad_funcs,variables,lc_wrt_args,qc_wrt_args,
                                    cp_wrt_args)
                                    
        # The function now returns an ADF object:
        return ADF(f, lc_wrt_vars, qc_wrt_vars, cp_wrt_vars)
    else:
#        try: # pythonic: fails gracefully when x is not an array-like object
#            return [atanh(xi) for xi in x]
#        except TypeError:
        if x.imag:
            return cmath.atanh(x)
        else:
            return math.atanh(x.real)

@_vectorize
def isinf(x):
    """
    Return True if the real or the imaginary part of x is positive or negative 
    infinity.
    """
#    try:
#        return [isinf(xi) for xi in x]
#    except TypeError:
    if isinstance(x, ADF):
        return isinf(x.x)
    else:
        if x.imag:
            return cmath.isinf(x)
        else:
            return math.isinf(x.real)
        
@_vectorize
def isnan(x):
    """
    Return True if the real or imaginary part of x is not a number (NaN).
    """
#    try:
#        return [isnan(xi) for xi in x]
#    except TypeError:
    if isinstance(x, ADF):
        return isnan(x.x)
    else:
        if x.imag:
            return cmath.isnan(x)
        else:
            return math.isnan(x.real)
 
@_vectorize
def phase(x):
    """
    Return the phase of x (also known as the argument of x).
    """
#    try:
#        return [atan2(xi.imag, xi.real) for xi in x]
#    except TypeError:
    return atan2(x.imag, x.real)

@_vectorize
def polar(x):
    """
    Return the representation of x in polar coordinates.
    """
    return (abs(x), phase(x))

@_vectorize
def rect(r, phi):
    """
    Return the complex number x with polar coordinates r and phi.
    """
#    try:
#        return [ri*(cos(phi_i) + sin(phi_i)*1j) for ri, phi_i in zip(r, phi)]
#    except TypeError:
#        try:
#            return [ri*(cos(phi) + sin(phi)*1j) for ri in r]
#        except TypeError:
#            try:
#                return [r*(cos(phi_i) + sin(phi_i)*1j) for phi_i in phi]
#            except TypeError:
    return r*(cos(phi) + sin(phi)*1j)

@_vectorize
def ceil(x):
    """
    Return the ceiling of x as a float, the smallest integer value greater than 
    or equal to x.
    """
    if isinstance(x,ADF):
        ad_funcs = list(map(to_auto_diff,[x]))

        x = ad_funcs[0].x
        
        ########################################
        # Nominal value of the constructed ADF:
        f = ceil(x)
        
        ########################################

        variables = ad_funcs[0]._get_variables(ad_funcs)
        
        if not variables or isinstance(f, bool):
            return f

        ########################################

        # Calculation of the derivatives with respect to the arguments
        # of f (ad_funcs):

        lc_wrt_args = [0.0]
        qc_wrt_args = [0.0]
        cp_wrt_args = 0.0

        ########################################
        # Calculation of the derivative of f with respect to all the
        # variables (Variable) involved.

        lc_wrt_vars,qc_wrt_vars,cp_wrt_vars = _apply_chain_rule(
                                    ad_funcs,variables,lc_wrt_args,qc_wrt_args,
                                    cp_wrt_args)
                                    
        # The function now returns an ADF object:
        return ADF(f, lc_wrt_vars, qc_wrt_vars, cp_wrt_vars)
    else:
#        try: # pythonic: fails gracefully when x is not an array-like object
#            return [ceil(xi) for xi in x]
#        except TypeError:
        return math.ceil(x)

@_vectorize
def cos(x):
    """
    Return the cosine of x radians.
    """
    if isinstance(x,ADF):
        ad_funcs = list(map(to_auto_diff,[x]))

        x = ad_funcs[0].x
        
        ########################################
        # Nominal value of the constructed ADF:
        f = cos(x)
        
        ########################################

        variables = ad_funcs[0]._get_variables(ad_funcs)
        
        if not variables or isinstance(f, bool):
            return f

        ########################################

        # Calculation of the derivatives with respect to the arguments
        # of f (ad_funcs):

        lc_wrt_args = [-sin(x)]
        qc_wrt_args = [-cos(x)]
        cp_wrt_args = 0.0

        ########################################
        # Calculation of the derivative of f with respect to all the
        # variables (Variable) involved.

        lc_wrt_vars,qc_wrt_vars,cp_wrt_vars = _apply_chain_rule(
                                    ad_funcs,variables,lc_wrt_args,qc_wrt_args,
                                    cp_wrt_args)
                                    
        # The function now returns an ADF object:
        return ADF(f, lc_wrt_vars, qc_wrt_vars, cp_wrt_vars)
    else:
#        try: # pythonic: fails gracefully when x is not an array-like object
#            return [cos(xi) for xi in x]
#        except TypeError:
        if x.imag:
            return cmath.cos(x)
        else:
            return math.cos(x.real)

@_vectorize
def cosh(x):
    """
    Return the hyperbolic cosine of x.
    """
    if isinstance(x,ADF):
        ad_funcs = list(map(to_auto_diff,[x]))

        x = ad_funcs[0].x
        
        ########################################
        # Nominal value of the constructed ADF:
        f = cosh(x)
        
        ########################################

        variables = ad_funcs[0]._get_variables(ad_funcs)
        
        if not variables or isinstance(f, bool):
            return f

        ########################################

        # Calculation of the derivatives with respect to the arguments
        # of f (ad_funcs):

        lc_wrt_args = [sinh(x)]
        qc_wrt_args = [cosh(x)]
        cp_wrt_args = 0.0

        ########################################
        # Calculation of the derivative of f with respect to all the
        # variables (Variable) involved.

        lc_wrt_vars,qc_wrt_vars,cp_wrt_vars = _apply_chain_rule(
                                    ad_funcs,variables,lc_wrt_args,qc_wrt_args,
                                    cp_wrt_args)
                                    
        # The function now returns an ADF object:
        return ADF(f, lc_wrt_vars, qc_wrt_vars, cp_wrt_vars)
    else:
#        try: # pythonic: fails gracefully when x is not an array-like object
#            return [cosh(xi) for xi in x]
#        except TypeError:
        if x.imag:
            return cmath.cosh(x)
        else:
            return math.cosh(x.real)

@_vectorize
def degrees(x):
    """
    Converts angle x from radians to degrees.
    """
    return (180/pi)*x

@_vectorize
def erf(x):
    """
    Return the error function at x.
    """
    if isinstance(x,ADF):
        ad_funcs = list(map(to_auto_diff,[x]))

        x = ad_funcs[0].x
        
        ########################################
        # Nominal value of the constructed ADF:
        f = erf(x)
        
        ########################################

        variables = ad_funcs[0]._get_variables(ad_funcs)
        
        if not variables or isinstance(f, bool):
            return f

        ########################################

        # Calculation of the derivatives with respect to the arguments
        # of f (ad_funcs):

        lc_wrt_args = [2*exp(-x**2)/sqrt(pi)]
        qc_wrt_args = [-4*x*exp(-x**2)/sqrt(pi)]
        cp_wrt_args = 0.0

        ########################################
        # Calculation of the derivative of f with respect to all the
        # variables (Variable) involved.

        lc_wrt_vars,qc_wrt_vars,cp_wrt_vars = _apply_chain_rule(
                                    ad_funcs,variables,lc_wrt_args,qc_wrt_args,
                                    cp_wrt_args)
                                    
        # The function now returns an ADF object:
        return ADF(f, lc_wrt_vars, qc_wrt_vars, cp_wrt_vars)
    else:
#        try: # pythonic: fails gracefully when x is not an array-like object
#            return [erf(xi) for xi in x]
#        except TypeError:
        return math.erf(x)

@_vectorize
def erfc(x):
    """
    Return the complementary error function at x.
    """
    return 1 - erf(x)
    
@_vectorize
def exp(x):
    """
    Return the exponential value of x
    """
    if isinstance(x,ADF):
        ad_funcs = list(map(to_auto_diff,[x]))

        x = ad_funcs[0].x
        
        ########################################
        # Nominal value of the constructed ADF:
        f = exp(x)
        
        ########################################

        variables = ad_funcs[0]._get_variables(ad_funcs)
        
        if not variables or isinstance(f, bool):
            return f

        ########################################

        # Calculation of the derivatives with respect to the arguments
        # of f (ad_funcs):

        lc_wrt_args = [exp(x)]
        qc_wrt_args = [exp(x)]
        cp_wrt_args = 0.0

        ########################################
        # Calculation of the derivative of f with respect to all the
        # variables (Variable) involved.

        lc_wrt_vars,qc_wrt_vars,cp_wrt_vars = _apply_chain_rule(
                                    ad_funcs,variables,lc_wrt_args,qc_wrt_args,
                                    cp_wrt_args)
                                    
        # The function now returns an ADF object:
        return ADF(f, lc_wrt_vars, qc_wrt_vars, cp_wrt_vars)
    else:
#        try: # pythonic: fails gracefully when x is not an array-like object
#            return [exp(xi) for xi in x]
#        except TypeError:
        if x.imag:
            return cmath.exp(x)
        else:
            return math.exp(x.real)

@_vectorize
def expm1(x):
    """
    Return e**x - 1. For small floats x, the subtraction in exp(x) - 1 can 
    result in a significant loss of precision; the expm1() function provides 
    a way to compute this quantity to full precision::

        >>> exp(1e-5) - 1  # gives result accurate to 11 places
        1.0000050000069649e-05
        >>> expm1(1e-5)    # result accurate to full precision
        1.0000050000166668e-05

    """
    if isinstance(x,ADF):
        ad_funcs = list(map(to_auto_diff,[x]))

        x = ad_funcs[0].x
        
        ########################################
        # Nominal value of the constructed ADF:
        f = expm1(x)
        
        ########################################

        variables = ad_funcs[0]._get_variables(ad_funcs)
        
        if not variables or isinstance(f, bool):
            return f

        ########################################

        # Calculation of the derivatives with respect to the arguments
        # of f (ad_funcs):

        lc_wrt_args = [exp(x)]
        qc_wrt_args = [exp(x)]
        cp_wrt_args = 0.0

        ########################################
        # Calculation of the derivative of f with respect to all the
        # variables (Variable) involved.

        lc_wrt_vars,qc_wrt_vars,cp_wrt_vars = _apply_chain_rule(
                                    ad_funcs,variables,lc_wrt_args,qc_wrt_args,
                                    cp_wrt_args)
                                    
        # The function now returns an ADF object:
        return ADF(f, lc_wrt_vars, qc_wrt_vars, cp_wrt_vars)
    else:
#        try: # pythonic: fails gracefully when x is not an array-like object
#            return [expm1(xi) for xi in x]
#        except TypeError:
        return math.expm1(x) 
    
@_vectorize
def fabs(x):
    """
    Return the absolute value of x.
    """
    if isinstance(x,ADF):
        ad_funcs = list(map(to_auto_diff,[x]))

        x = ad_funcs[0].x
        
        ########################################
        # Nominal value of the constructed ADF:
        f = fabs(x)
        
        ########################################

        variables = ad_funcs[0]._get_variables(ad_funcs)
        
        if not variables or isinstance(f, bool):
            return f

        ########################################

        # Calculation of the derivatives with respect to the arguments
        # of f (ad_funcs):

        # catch the x=0 exception
        try:
            lc_wrt_args = [x/fabs(x)]
            qc_wrt_args = [1/fabs(x)-(x**2)/fabs(x)**3]
        except ZeroDivisionError:
            lc_wrt_args = [0.0]
            qc_wrt_args = [0.0]
            
        cp_wrt_args = 0.0

        ########################################
        # Calculation of the derivative of f with respect to all the
        # variables (Variable) involved.
        lc_wrt_vars, qc_wrt_vars, cp_wrt_vars = _apply_chain_rule(
                                    ad_funcs, variables, lc_wrt_args,
                                    qc_wrt_args, cp_wrt_args)
                                    
                                    
        # The function now returns an ADF object:
        return ADF(f, lc_wrt_vars, qc_wrt_vars, cp_wrt_vars)
    else:
#        try: # pythonic: fails gracefully when x is not an array-like object
#            return [fabs(xi) for xi in x]
#        except TypeError:
        return math.fabs(x) 

@_vectorize
def factorial(x):
    """
    Return x factorial. Uses the relationship factorial(x)==gamma(x+1) to 
    calculate derivatives.
    """
    return gamma(x+1)

@_vectorize
def floor(x):
    """
    Return the floor of x as a float, the largest integer value less than or 
    equal to x.
    """
    if isinstance(x,ADF):
        ad_funcs = list(map(to_auto_diff,[x]))

        x = ad_funcs[0].x
        
        ########################################
        # Nominal value of the constructed ADF:
        f = floor(x)
        
        ########################################

        variables = ad_funcs[0]._get_variables(ad_funcs)
        
        if not variables or isinstance(f, bool):
            return f

        ########################################

        # Calculation of the derivatives with respect to the arguments
        # of f (ad_funcs):

        lc_wrt_args = [0.0]
        qc_wrt_args = [0.0]
        cp_wrt_args = 0.0

        ########################################
        # Calculation of the derivative of f with respect to all the
        # variables (Variable) involved.

        lc_wrt_vars,qc_wrt_vars,cp_wrt_vars = _apply_chain_rule(
                                    ad_funcs,variables,lc_wrt_args,qc_wrt_args,
                                    cp_wrt_args)
                                    
        # The function now returns an ADF object:
        return ADF(f, lc_wrt_vars, qc_wrt_vars, cp_wrt_vars)
    else:
#        try: # pythonic: fails gracefully when x is not an array-like object
#            return [floor(xi) for xi in x]
#        except TypeError:
        return math.floor(x)

@_vectorize
def gamma(x):
    """
    Return the Gamma function at x (uses the Lanczos approximation).
    """
    # This is the Lanczos approximation of the Gamma function, as copied 
    # from the Wikipedia article: 
    #    http://en.wikipedia.org/wiki/Lanczos_approximation
    # It is designed to work with real and complex arguments
    # Coefficients used by the GNU Scientific Library
    
#    # 9-term approximation
#    g = 7
#    p = [0.99999999999980993, 
#         676.5203681218851, 
#         -1259.1392167224028,
#         771.32342877765313, 
#         -176.61502916214059, 
#         12.507343278686905,
#         -0.13857109526572012, 
#         9.9843695780195716e-6, 
#         1.5056327351493116e-7]
    
    # 15-term approximation
    g = 607./128
    p = [0.99999999999999709182,
         57.156235665862923517,
         -59.597960355475491248,
         14.136097974741747174,
         -0.49191381609762019978,
         .33994649984811888699e-4,
         .46523628927048575665e-4,
         -.98374475304879564677e-4,
         .15808870322491248884e-3,
         -.21026444172410488319e-3,
         .21743961811521264320e-3,
         -.16431810653676389022e-3,
         .84418223983852743293e-4,
         -.26190838401581408670e-4,
         .36899182659531622704e-5]
         
    # Reflection formula
    if x.real < 0.5:
        return pi/(sin(pi*x)*gamma(1 - x))
    else:
        x -= 1
        z = p[0]
        for i in range(1, len(p)):
            z += p[i]/(x + i)
        t = x + g + 0.5
        _gamma = sqrt(2*pi)*t**(x + 0.5)*exp(-t)*z
        if isinstance(x, ADF):
            if _gamma.imag:
                return _gamma
            else:
                return _gamma.toFloat()
        else:
            if _gamma.imag:
                return _gamma
            else:
                return _gamma.real


@_vectorize
def hypot(x,y):
    """
    Return the Euclidean norm, ``sqrt(x*x + y*y)``. This is the length of the 
    vector from the origin to point ``(x, y)``.
    """
    return sqrt(x*x+y*y)
    
@_vectorize
def lgamma(x):
    """
    Return the natural logarithm of the absolute value of the Gamma function 
    at x.
    """
    return log(abs(gamma(x)))
    
@_vectorize
def log(x, base=None):
    """
    With one argument, return the natural logarithm of x (to base e).

    With two arguments, return the logarithm of x to the given base, calculated 
    as ``log(x)/log(base)``.
    """
    if base is None:
        return log(x, base=e)
    
    if isinstance(x,ADF):
        
        ad_funcs = list(map(to_auto_diff,[x]))

        x = ad_funcs[0].x
        
        ########################################
        # Nominal value of the constructed ADF:
        f = log(x, base)
        
        ########################################

        variables = ad_funcs[0]._get_variables(ad_funcs)
        
        if not variables or isinstance(f, bool):
            return f

        ########################################

        # Calculation of the derivatives with respect to the arguments
        # of f (ad_funcs):

        lc_wrt_args = [1./(x * ln(base))]
        qc_wrt_args = [-1./(x**2 * ln(base))]
        cp_wrt_args = 0.0

        ########################################
        # Calculation of the derivative of f with respect to all the
        # variables (Variable) involved.

        lc_wrt_vars,qc_wrt_vars,cp_wrt_vars = _apply_chain_rule(
                                    ad_funcs,variables,lc_wrt_args,qc_wrt_args,
                                    cp_wrt_args)
                                    
        # The function now returns an ADF object:
        return ADF(f, lc_wrt_vars, qc_wrt_vars, cp_wrt_vars)
    
    else:
#        try: # pythonic: fails gracefully when x is not an array-like object
#            return [log(xi) for xi in x]
#        except TypeError:
        if x.imag:
            return cmath.log(x, base)
        else:
            return math.log(x.real, base)

@_vectorize
def log10(x):
    """
    Return the base-10 logarithm of x. This is usually more accurate than 
    ``log(x, 10)``.
    """
    if isinstance(x,ADF):
        
        ad_funcs = list(map(to_auto_diff,[x]))

        x = ad_funcs[0].x
        
        ########################################
        # Nominal value of the constructed ADF:
        f = log10(x)
        
        ########################################

        variables = ad_funcs[0]._get_variables(ad_funcs)
        
        if not variables or isinstance(f, bool):
            return f

        ########################################

        # Calculation of the derivatives with respect to the arguments
        # of f (ad_funcs):

        lc_wrt_args = [1/x/log(10)]
        qc_wrt_args = [-1./x**2/log(10)]
        cp_wrt_args = 0.0

        ########################################
        # Calculation of the derivative of f with respect to all the
        # variables (Variable) involved.

        lc_wrt_vars,qc_wrt_vars,cp_wrt_vars = _apply_chain_rule(
                                    ad_funcs,variables,lc_wrt_args,qc_wrt_args,
                                    cp_wrt_args)
                                    
        # The function now returns an ADF object:
        return ADF(f, lc_wrt_vars, qc_wrt_vars, cp_wrt_vars)
    
    else:
#        try: # pythonic: fails gracefully when x is not an array-like object
#            return [log10(xi) for xi in x]
#        except TypeError:
        if x.imag:
            return cmath.log10(x)
        else:
            return math.log10(x.real)

@_vectorize
def log1p(x):
    """
    Return the base-10 logarithm of x. This is usually more accurate than 
    ``log(x, 10)``.
    """
    if isinstance(x,ADF):
        
        ad_funcs = list(map(to_auto_diff,[x]))

        x = ad_funcs[0].x
        
        ########################################
        # Nominal value of the constructed ADF:
        f = log1p(x)
        
        ########################################

        variables = ad_funcs[0]._get_variables(ad_funcs)
        
        if not variables or isinstance(f, bool):
            return f

        ########################################

        # Calculation of the derivatives with respect to the arguments
        # of f (ad_funcs):

        lc_wrt_args = [1/(x+1)]
        qc_wrt_args = [-1./(x+1)**2]
        cp_wrt_args = 0.0

        ########################################
        # Calculation of the derivative of f with respect to all the
        # variables (Variable) involved.

        lc_wrt_vars,qc_wrt_vars,cp_wrt_vars = _apply_chain_rule(
                                    ad_funcs,variables,lc_wrt_args,qc_wrt_args,
                                    cp_wrt_args)
                                    
        # The function now returns an ADF object:
        return ADF(f, lc_wrt_vars, qc_wrt_vars, cp_wrt_vars)
    
    else:
#        try: # pythonic: fails gracefully when x is not an array-like object
#            return [log1p(xi) for xi in x]
#        except TypeError:
        return math.log1p(x)

@_vectorize
def pow(x, y):
    """
    Return x raised to the power y. 
    """
    return x**y

@_vectorize
def radians(x):
    """
    Converts angle x from degrees to radians.
    """
    return (pi/180)*x
    
@_vectorize
def sin(x):
    """
    Return the sine of x, in radians.
    """
    if isinstance(x,ADF):
        ad_funcs = list(map(to_auto_diff,[x]))

        x = ad_funcs[0].x
        
        ########################################
        # Nominal value of the constructed ADF:
        f = sin(x)
        
        ########################################

        variables = ad_funcs[0]._get_variables(ad_funcs)
        
        if not variables or isinstance(f, bool):
            return f

        ########################################

        # Calculation of the derivatives with respect to the arguments
        # of f (ad_funcs):

        lc_wrt_args = [cos(x)]
        qc_wrt_args = [-sin(x)]
        cp_wrt_args = 0.0

        ########################################
        # Calculation of the derivative of f with respect to all the
        # variables (Variable) involved.

        lc_wrt_vars,qc_wrt_vars,cp_wrt_vars = _apply_chain_rule(
                                    ad_funcs,variables,lc_wrt_args,qc_wrt_args,
                                    cp_wrt_args)
                                    
        # The function now returns an ADF object:
        return ADF(f, lc_wrt_vars, qc_wrt_vars, cp_wrt_vars)
    else:
#        try: # pythonic: fails gracefully when x is not an array-like object
#            return [sin(xi) for xi in x]
#        except TypeError:
        if x.imag:
            return cmath.sin(x)
        else:
            return math.sin(x.real)

@_vectorize
def sinh(x):
    """
    Return the hyperbolic sine of x.
    """
    if isinstance(x,ADF):
        ad_funcs = list(map(to_auto_diff,[x]))

        x = ad_funcs[0].x
        
        ########################################
        # Nominal value of the constructed ADF:
        f = sinh(x)
        
        ########################################

        variables = ad_funcs[0]._get_variables(ad_funcs)
        
        if not variables or isinstance(f, bool):
            return f

        ########################################

        # Calculation of the derivatives with respect to the arguments
        # of f (ad_funcs):

        lc_wrt_args = [cosh(x)]
        qc_wrt_args = [sinh(x)]
        cp_wrt_args = 0.0

        ########################################
        # Calculation of the derivative of f with respect to all the
        # variables (Variable) involved.

        lc_wrt_vars,qc_wrt_vars,cp_wrt_vars = _apply_chain_rule(
                                    ad_funcs,variables,lc_wrt_args,qc_wrt_args,
                                    cp_wrt_args)
                                    
        # The function now returns an ADF object:
        return ADF(f, lc_wrt_vars, qc_wrt_vars, cp_wrt_vars)
    else:
#        try: # pythonic: fails gracefully when x is not an array-like object
#            return [sinh(xi) for xi in x]
#        except TypeError:
        if x.imag:
            return cmath.sinh(x)
        else:
            return math.sinh(x.real)

@_vectorize
def sqrt(x):
    """
    Return the square root of x.
    """
#    try:
#        return [xi**0.5 for xi in x]
#    except TypeError:
    return x**0.5
            
@_vectorize
def tan(x):
    """
    Return the tangent of x, in radians.
    """
    if isinstance(x,ADF):
        ad_funcs = list(map(to_auto_diff,[x]))

        x = ad_funcs[0].x
        
        ########################################
        # Nominal value of the constructed ADF:
        f = tan(x)
        
        ########################################

        variables = ad_funcs[0]._get_variables(ad_funcs)
        
        if not variables or isinstance(f, bool):
            return f

        ########################################

        # Calculation of the derivatives with respect to the arguments
        # of f (ad_funcs):

        lc_wrt_args = [1./(cos(x))**2]
        qc_wrt_args = [2*sin(x)/(cos(x))**3]
        cp_wrt_args = 0.0

        ########################################
        # Calculation of the derivative of f with respect to all the
        # variables (Variable) involved.

        lc_wrt_vars,qc_wrt_vars,cp_wrt_vars = _apply_chain_rule(
                                    ad_funcs,variables,lc_wrt_args,qc_wrt_args,
                                    cp_wrt_args)
                                    
        # The function now returns an ADF object:
        return ADF(f, lc_wrt_vars, qc_wrt_vars, cp_wrt_vars)
    else:
#        try: # pythonic: fails gracefully when x is not an array-like object
#            return [tan(xi) for xi in x]
#        except TypeError:
        if x.imag:
            return cmath.tan(x)
        else:
            return math.tan(x.real)

@_vectorize
def tanh(x):
    """
    Return the hyperbolic tangent of x.
    """
    if isinstance(x,ADF):
        ad_funcs = list(map(to_auto_diff,[x]))

        x = ad_funcs[0].x
        
        ########################################
        # Nominal value of the constructed ADF:
        f = tanh(x)
        
        ########################################

        variables = ad_funcs[0]._get_variables(ad_funcs)
        
        if not variables or isinstance(f, bool):
            return f

        ########################################

        # Calculation of the derivatives with respect to the arguments
        # of f (ad_funcs):

        lc_wrt_args = [1./(cosh(x))**2]
        qc_wrt_args = [-2*sinh(x)/(cosh(x))**3]
        cp_wrt_args = 0.0

        ########################################
        # Calculation of the derivative of f with respect to all the
        # variables (Variable) involved.

        lc_wrt_vars,qc_wrt_vars,cp_wrt_vars = _apply_chain_rule(
                                    ad_funcs,variables,lc_wrt_args,qc_wrt_args,
                                    cp_wrt_args)
                                    
        # The function now returns an ADF object:
        return ADF(f, lc_wrt_vars, qc_wrt_vars, cp_wrt_vars)
    else:
#        try: # pythonic: fails gracefully when x is not an array-like object
#            return [tanh(xi) for xi in x]
#        except TypeError:
        if x.imag:
            return cmath.tanh(x)
        else:
            return math.tanh(x.real)

@_vectorize
def trunc(x):
    """
    Return the **Real** value x truncated to an **Integral** (usually a 
    long integer). Uses the ``__trunc__`` method.
    """
    if isinstance(x,ADF):
        ad_funcs = list(map(to_auto_diff,[x]))

        x = ad_funcs[0].x
        
        ########################################
        # Nominal value of the constructed ADF:
        f = trunc(x)
        
        ########################################

        variables = ad_funcs[0]._get_variables(ad_funcs)
        
        if not variables or isinstance(f, bool):
            return f

        ########################################

        # Calculation of the derivatives with respect to the arguments
        # of f (ad_funcs):

        lc_wrt_args = [0.0]
        qc_wrt_args = [0.0]
        cp_wrt_args = 0.0

        ########################################
        # Calculation of the derivative of f with respect to all the
        # variables (Variable) involved.

        lc_wrt_vars,qc_wrt_vars,cp_wrt_vars = _apply_chain_rule(
                                    ad_funcs,variables,lc_wrt_args,qc_wrt_args,
                                    cp_wrt_args)
                                    
        # The function now returns an ADF object:
        return ADF(f, lc_wrt_vars, qc_wrt_vars, cp_wrt_vars)
    else:
        try: # pythonic: fails gracefully when x is not an array-like object
            return [trunc(xi) for xi in x]
        except TypeError:
            return math.trunc(x)

### OTHER CONVENIENCE FUNCTIONS ###############################################

@_vectorize
def csc(x):
    """
    Return the cosecant of x.
    """
    return 1.0/sin(x)
    
@_vectorize
def sec(x):
    """
    Return the secant of x.
    """
    return 1.0/cos(x)
    
@_vectorize
def cot(x):
    """
    Return the cotangent of x.
    """
    return 1.0/tan(x)

@_vectorize
def csch(x):
    """
    Return the hyperbolic cosecant of x.
    """
    return 1.0/sinh(x)
    
@_vectorize
def sech(x):
    """
    Return the hyperbolic secant of x.
    """
    return 1.0/cosh(x)
    
@_vectorize
def coth(x):
    """
    Return the hyperbolic cotangent of x.
    """
    return 1.0/tanh(x)

@_vectorize
def acsc(x):
    """
    Return the inverse cosecant of x.
    """
    return asin(1.0/x)

@_vectorize
def asec(x):
    """
    Return the inverse secant of x.
    """
    return acos(1.0/x)
    
@_vectorize
def acot(x):
    """
    Return the inverse cotangent of x.
    """
    return atan(1.0/x)

@_vectorize
def acsch(x):
    """
    Return the inverse hyperbolic cosecant of x.
    """
    return asinh(1.0/x)

@_vectorize
def asech(x):
    """
    Return the inverse hyperbolic secant of x.
    """
    return acosh(1.0/x)
    
@_vectorize
def acoth(x):
    """
    Return the inverse hyperbolic cotangent of x.
    """
    return atanh(1.0/x)

@_vectorize
def ln(x):
    """
    Return the natural logarithm of x.
    """
    return log(x)
