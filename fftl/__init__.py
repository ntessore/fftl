# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
'''
:mod:`fftl` --- Generalised FFTLog
==================================

.. currentmodule:: fftl

The functionality of the :mod:`fftl` module is provided by the
:func:`fftl` routine to compute the generalised FFTLog integral
transform for a given kernel.

.. autofunction:: fftl.fftl

.. autodecorator:: fftl.fftl.wrap

'''

__version__ = '2023.6'

__all__ = [
    'fftl',
]

from importlib import import_module
from inspect import signature


def fftl(u, r, ar, *, q=0.0, kr=1.0, low_ringing=True, deriv=False,
         xp='numpy'):
    r'''Generalised FFTLog for integral transforms.

    Computes integral transforms for arbitrary kernels using a generalisation
    of Hamilton's method [1]_ for the FFTLog algorithm [2]_.

    The kernel of the integral transform is characterised by the coefficient
    function ``u``, see notes below, which must be callable and accept complex
    input arrays.  Additional arguments for ``u`` can be passed with the
    optional ``args`` parameter.

    The function to be transformed must be given on a logarithmic grid ``r``.
    The result of the integral transform is similarly computed on a logarithmic
    grid ``k = kr/r``, where ``kr`` is a scalar constant (default: 1) which
    shifts the logarithmic output grid.  The selected value of ``kr`` is
    automatically changed to the nearest low-ringing value if ``krgood`` is
    true (the default).

    The integral transform can optionally be biased, see notes below.

    The function can optionally at the same time return the derivative of the
    integral transform with respect to the logarithm of ``k``, by setting
    ``deriv`` to true.

    Parameters
    ----------
    u : callable
        Coefficient function.  Must have signature ``u(x, *args)`` and support
        complex input arrays.
    r : array_like (N,)
        Grid of input points.  Must have logarithmic spacing.
    ar : array_like (..., N)
        Function values.  If multidimensional, the integral transform applies
        to the last axis, which must agree with input grid.
    q : float, optional
        Bias parameter for integral transform.
    kr : float, optional
        Shift parameter for logarithmic output grid.
    low_ringing : bool, optional
        Change given ``kr`` to the nearest value fulfilling the low-ringing
        condition.
    deriv : bool, optional
        Also return the first derivative of the integral transform.

    Returns
    -------
    k : array_like (N,)
        Grid of output points.
    ak : array_like (..., N)
        Integral transform evaluated at ``k``.
    dak : array_like (..., N), optional
        If ``deriv`` is true, the derivative of ``ak`` with respect to the
        logarithm of ``k``.

    Other Parameters
    ----------------
    xp : str or object
        An optional namespace that contains array functions such as
        ``xp.log()``, ``xp.exp()``, etc.  If a string, a module with
        that name will be imported.

    Notes
    -----
    Computes integral transforms of the form

    .. math::

        \tilde{a}(k) = \int_{0}^{\infty} \! a(r) \, T(kr) \, dr

    for arbitrary kernels :math:`T`.

    If :math:`a(r)` is given on a logarithmic grid of :math:`r` values, the
    integral transform can be computed for a logarithmic grid of :math:`k`
    values with a modification of Hamilton's FFTLog algorithm,

    .. math::

        U(x) = \int_{0}^{\infty} \! t^x \, T(t) \, dt \;.

    The generalised FFTLog algorithm therefore only requires the coefficient
    function :math:`U` for the given kernel.  Everything else, and in
    particular how to construct a well-defined transform, remains exactly the
    same as in Hamilton's original algorithm.

    The transform can optionally be biased,

    .. math::

        \tilde{a}(k) = k^{-q} \int_{0}^{\infty} \! [a(r) \, r^{-q}] \,
                                                    [T(kr) \, (kr)^q] \, dr \;,

    where :math:`q` is the bias parameter.  The respective biasing factors
    :math:`r^{-q}` and :math:`k^{-q}` for the input and output values are
    applied internally.

    References
    ----------
    .. [1] Hamilton A. J. S., 2000, MNRAS, 312, 257 (astro-ph/9905191)
    .. [2] Talman J. D., 1978, J. Comp. Phys., 29, 35

    Examples
    --------
    Compute the one-sided Laplace transform of the hyperbolic tangent function.
    The kernel of the Laplace transform is :math:`\exp(-kt)`, which determines
    the coefficient function.

    >>> import numpy as np
    >>> from scipy.special import gamma, digamma
    >>>
    >>> def u_laplace(x):
    ...     # requires Re(x) = q > -1
    ...     return gamma(1 + x)

    Create the input function values on a logarithmic grid.

    >>> r = np.logspace(-4, 4, 100)
    >>> ar = np.tanh(r)
    >>>
    >>> import matplotlib.pyplot as plt
    >>> plt.loglog(r, ar)
    >>> plt.xlabel('$r$')
    >>> plt.ylabel('$\\tanh(r)$')
    >>> plt.show()

    Compute the Laplace transform, and compare with the analytical result.

    >>> from fftl import fftl
    >>>
    >>> k, ak = fftl(u_laplace, r, ar)
    >>>
    >>> lt = (digamma((k+2)/4) - digamma(k/4) - 2/k)/2
    >>>
    >>> plt.loglog(k, ak)
    >>> plt.loglog(k, lt, ':')
    >>> plt.xlabel('$k$')
    >>> plt.ylabel('$L[\\tanh](k)$')
    >>> plt.show()

    The numerical Laplace transform has an issue on the right, which is due to
    the circular nature of the FFTLog integral transform.  The effect is
    mitigated by computing a biased transform with ``q = 0.5``.  Good values of
    the bias parameter ``q`` depend on the shape of the input function.

    >>> k, ak = fftl(u_laplace, r, ar, q=0.5)
    >>>
    >>> plt.loglog(k, ak)
    >>> plt.loglog(k, lt, ':')
    >>> plt.xlabel('$k$')
    >>> plt.ylabel('$L[\\tanh](k)$')
    >>> plt.show()

    '''

    # import the array namespace if module name is given
    if isinstance(xp, str):
        xp = import_module(xp)

    if r.ndim != 1:
        raise TypeError('r must be 1d array')
    if ar.shape[-1] != r.shape[-1]:
        raise TypeError('last axis of ar must agree with r')

    # inputs
    n, = r.shape

    # log spacing
    dlnr = xp.log(r[-1]/r[0])/(n-1)

    # make sure given r is logarithmic grid
    if xp.any(xp.abs(xp.log(r[1:]/r[:-1]) - dlnr) > 1e-10*xp.abs(dlnr)):
        raise ValueError('r it not a logarithmic grid')

    # get shift parameter, with or without low-ringing condition
    if low_ringing:
        _lnkr = xp.log(kr)
        _y = xp.pi/dlnr
        _um = xp.exp(-1j*_y*_lnkr)*u(q + 1j*_y)
        _a = xp.angle(_um)/xp.pi
        lnkr = _lnkr + dlnr*(_a - xp.round(_a))
    else:
        lnkr = xp.log(kr)

    # transform factor
    y = xp.linspace(0, 2*xp.pi*(n//2)/(n*dlnr), n//2+1)
    um = xp.exp(-1j*y*lnkr)*u(q + 1j*y)

    # low-ringing kr should make last coefficient real
    if low_ringing and xp.any(xp.abs(um[-1].imag) > 1e-15):
        raise ValueError('unable to construct low-ringing transform, '
                         'try odd number of points or different q')

    # fix last coefficient to real when n is even
    if not n & 1:
        um.imag[-1] = 0

    # bias input
    if q != 0:
        ar = ar*r**(-q)

    # set up k in log space
    k = xp.exp(lnkr)/r[::-1]

    # transform via real FFT
    cm = xp.fft.rfft(ar, axis=-1)
    cm *= um
    ak = xp.fft.irfft(cm, n, axis=-1)
    ak[..., :] = ak[..., ::-1]

    # debias output
    ak /= k**(1+q)

    # output grid and transform
    result = (k, ak)

    # derivative
    if deriv:
        cm *= -(1 + q + 1j*y)
        dak = xp.fft.irfft(cm, n, axis=-1)
        dak[..., :] = dak[..., ::-1]
        dak /= k**(1+q)
        result = result + (dak,)

    # return chosen outputs
    return result


def _fftl_wrap(func):
    '''Decorator for functions that wrap :func:`fftl`.

    Gives wrappers the correct signature and default values for keyword
    parameters.

    Examples
    --------
    Create a custom :func:`fftl` function that applies the transform to
    ``ar*r`` instead of ``ar` and changes the default value of ``kr``:

    >>> from fftl import fftl
    >>> @fftl.wrap
    ... def myfftl(u, r, ar, *, q, kr=0.5, **kwargs):
    ...     # default values for q and kr are set by wrap
    ...     print(f'parameters: {q=}, {kr=}')
    ...     return fftl(u, r, ar*r, q=q, kr=kr, **kwargs)
    ...
    >>> from inspect import signature
    >>> signature(myfftl)
    <Signature (u, r, ar, *, q=0.0, kr=0.5, low_ringing=True, deriv=False, xp='numpy')>  # noqa: E501

    '''

    fftl_sig = signature(fftl)
    func_sig = signature(func)

    parameters = []
    kwdefaults = {}

    for par in func_sig.parameters.values():
        if (par.kind is par.KEYWORD_ONLY and par.default is par.empty
                and fftl_sig.parameters[par.name].default is not par.empty):
            default = fftl_sig.parameters[par.name].default
            parameters.append(par.replace(default=default))
            kwdefaults[par.name] = default
        elif par.kind is par.VAR_KEYWORD:
            for copy_par in fftl_sig.parameters.values():
                if (copy_par.kind is copy_par.KEYWORD_ONLY
                        and copy_par.default is not copy_par.empty
                        and copy_par.name not in func_sig.parameters):
                    parameters.append(copy_par)
        else:
            parameters.append(par)

    func.__signature__ = func_sig.replace(parameters=parameters)

    if kwdefaults:
        if func.__kwdefaults__ is None:
            func.__kwdefaults__ = kwdefaults
        else:
            func.__kwdefaults__.update(kwdefaults)

    if not func.__doc__:
        func.__doc__ = fftl.__doc__

    return func


# make available as @fftl.wrap on the fftl function
_fftl_wrap.__name__ = 'wrap'
_fftl_wrap.__qualname__ = 'fftl.wrap'
fftl.wrap = _fftl_wrap
