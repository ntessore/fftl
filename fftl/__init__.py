# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
"""
:mod:`fftl` --- Generalised FFTLog
==================================

.. currentmodule:: fftl

The functionality of the :mod:`fftl` module is provided by the
:func:`fftl` routine to compute the generalised FFTLog integral
transform for a given kernel.

.. autofunction:: fftl

.. autodecorator:: transform

"""

from __future__ import annotations

__version__ = "2025.1"

__all__ = [
    "fftl",
    "transform",
]

from importlib import import_module
from inspect import signature


def fftl(u, r, ar, *, q=0.0, kr=1.0, low_ringing=True, deriv=False, xp="numpy"):
    r"""Generalised FFTLog for integral transforms.

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
    >>> plt.loglog(r, ar)                                   # doctest: +SKIP
    >>> plt.xlabel('$r$')                                   # doctest: +SKIP
    >>> plt.ylabel('$\\tanh(r)$')                           # doctest: +SKIP
    >>> plt.show()

    Compute the Laplace transform, and compare with the analytical result.

    >>> import fftl
    >>>
    >>> k, ak = fftl.fftl(u_laplace, r, ar)
    >>>
    >>> lt = (digamma((k+2)/4) - digamma(k/4) - 2/k)/2
    >>>
    >>> plt.loglog(k, ak)                                   # doctest: +SKIP
    >>> plt.loglog(k, lt, ':')                              # doctest: +SKIP
    >>> plt.xlabel('$k$')                                   # doctest: +SKIP
    >>> plt.ylabel('$L[\\tanh](k)$')                        # doctest: +SKIP
    >>> plt.show()

    The numerical Laplace transform has an issue on the right, which is due to
    the circular nature of the FFTLog integral transform.  The effect is
    mitigated by computing a biased transform with ``q = 0.5``.  Good values of
    the bias parameter ``q`` depend on the shape of the input function.

    >>> k, ak = fftl.fftl(u_laplace, r, ar, q=0.5)
    >>>
    >>> plt.loglog(k, ak)                                   # doctest: +SKIP
    >>> plt.loglog(k, lt, ':')                              # doctest: +SKIP
    >>> plt.xlabel('$k$')                                   # doctest: +SKIP
    >>> plt.ylabel('$L[\\tanh](k)$')                        # doctest: +SKIP
    >>> plt.show()

    """

    # import the array namespace if module name is given
    if isinstance(xp, str):
        xp = import_module(xp)

    if r.ndim != 1:
        raise TypeError("r must be 1d array")
    if ar.shape[-1] != r.shape[-1]:
        raise TypeError("last axis of ar must agree with r")

    # input size
    N = r.size

    # log spacing
    L = xp.log(r[-1] / r[0])

    # make sure given r is logarithmic grid
    if xp.any(xp.abs(xp.log(r[1:] / r[:-1]) * (N - 1) - L) > 1e-10 * xp.abs(L)):
        raise ValueError("r is not a logarithmic grid")

    # frequencies of real FFT
    y = 2 * xp.pi / L * xp.arange(N // 2 + 1)

    # get logarithmic shift
    lnkr = xp.log(kr)

    # transform factor
    um = xp.exp(-1j * y * lnkr) * u(q + 1j * y)

    # low-ringing condition to make u_{N/2} real
    if low_ringing:
        if N % 2 == 0:
            y_nhalf = y[-1]
            um_nhalf = um[-1]
        else:
            y_nhalf = 2 * xp.pi / L * (N / 2)
            um_nhalf = xp.exp(-1j * y_nhalf * lnkr) * u(q + 1j * y_nhalf)
        if um_nhalf.imag != 0.0:
            a = xp.angle(um_nhalf)
            delt = (a - xp.round(a / xp.pi) * xp.pi) / y_nhalf
            lnkr += delt
            um *= xp.exp(-1j * y * delt)

    # fix last coefficient to real when N is even
    if N % 2 == 0:
        um.imag[-1] = 0

    # bias input
    if q != 0:
        ar = ar * r ** (-q)

    # set up k in log space
    k = xp.exp(lnkr) / r[::-1]

    # transform via real FFT
    cm = xp.fft.rfft(ar, axis=-1)
    cm *= um
    ak = xp.fft.irfft(cm, N, axis=-1)
    ak[..., :] = ak[..., ::-1]

    # debias output
    ak /= k ** (1 + q)

    # output grid and transform
    result = (k, ak)

    # derivative
    if deriv:
        cm *= -(1 + q + 1j * y)
        dak = xp.fft.irfft(cm, N, axis=-1)
        dak[..., :] = dak[..., ::-1]
        dak /= k ** (1 + q)
        result = result + (dak,)

    # return chosen outputs
    return result


def transform(wrapper):
    """Decorator for functions that wrap :func:`fftl`.

    Gives wrappers the correct signature and default values for keyword
    parameters.

    Examples
    --------
    Create a custom :func:`fftl` function that applies the transform to
    ``ar*r`` instead of ``ar`` and changes the default value of ``kr``:

    >>> import fftl
    >>> @fftl.transform
    ... def myfftl(u, r, ar, *, q, kr=0.5, **kwargs):
    ...     # default values for q and kr are set by wrap
    ...     print(f'parameters: q={q}, kr={kr}')
    ...     return fftl.fftl(u, r, ar*r, q=q, kr=kr, **kwargs)
    ...
    >>> from inspect import signature
    >>> signature(myfftl)  # doctest: +ELLIPSIS
    <Signature (u, r, ar, *, q=0.0, kr=0.5, low_ringing=True, ...)>

    """

    fftl_sig = signature(fftl)
    fftl_par = fftl_sig.parameters

    wrapper_sig = signature(wrapper)
    wrapper_par = wrapper_sig.parameters

    parameters = []
    kwdefaults = {}

    for par in wrapper_par.values():
        if (
            par.kind is par.KEYWORD_ONLY
            and par.default is par.empty
            and fftl_par[par.name].default is not par.empty
        ):
            default = fftl_par[par.name].default
            kwdefaults[par.name] = default
            parameters.append(par.replace(default=default))
        elif par.kind is par.VAR_KEYWORD:
            parameters.extend(
                _par
                for _par in fftl_par.values()
                if _par.kind is _par.KEYWORD_ONLY
                and _par.default is not _par.empty
                and _par.name not in wrapper_par
            )
        else:
            parameters.append(par)

    wrapper.__signature__ = wrapper_sig.replace(parameters=parameters)

    if kwdefaults:
        if wrapper.__kwdefaults__ is None:
            wrapper.__kwdefaults__ = kwdefaults
        else:
            wrapper.__kwdefaults__.update(kwdefaults)

    if not wrapper.__doc__:
        wrapper.__doc__ = fftl.__doc__

    return wrapper


def requires(
    _lo: float | None = None,
    _up: float | None = None,
    /,
    **values,
) -> None:
    """
    Check that *value* lies in the open interval between *lower* and *upper*.
    """
    for name, value in values.items():
        if _lo is not None and not value > _lo:
            raise ValueError(f"expected {name} > {_lo}, got {value}")
        if _up is not None and not value < _up:
            raise ValueError(f"expected {name} < {_up}, got {value}")
