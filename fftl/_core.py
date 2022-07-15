# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
'''internal module for core functionality'''

import numpy as np

PI = np.pi


def fftl(u, r, ar, q=0.0, args=(), kr=1.0, krgood=True, deriv=False):
    r'''generalised FFTLog for integral transforms

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
    args : tuple, optional
        Additional arguments for the coefficient function ``u``.
    kr : float, optional
        Shift parameter for logarithmic output grid.
    krgood : bool, optional
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

    if np.ndim(r) != 1:
        raise TypeError('r must be 1d array')
    if np.shape(ar)[-1] != len(r):
        raise TypeError('last axis of ar must agree with r')

    # inputs
    n = len(r)
    lnkr = np.log(kr)

    # make sure given r is logarithmic grid
    if not np.allclose(r, np.geomspace(r[0], r[-1], n)):
        raise ValueError('r it not a logarithmic grid')

    # log spacing
    dlnr = (np.log(r[-1]) - np.log(r[0]))/(n-1)

    # low-ringing condition
    if krgood:
        y = PI/dlnr
        um = np.exp(-1j*y*lnkr)*u(q + 1j*y, *args)
        a = np.angle(um)/PI
        lnkr = lnkr + dlnr*(a - np.round(a))

    # transform factor
    y = np.linspace(0, 2*PI*(n//2)/(n*dlnr), n//2+1)
    um = np.exp(-1j*y*lnkr)*u(q + 1j*y, *args)

    # low-ringing kr should make last coefficient real
    if krgood and not np.isclose(um[-1].imag, 0):
        raise ValueError('unable to construct low-ringing transform, '
                         'try odd number of points or different q')

    # fix last coefficient to real when n is even
    if not n & 1:
        um.imag[-1] = 0

    # bias input
    if q != 0:
        ar = ar*r**(-q)

    # set up k in log space
    k = np.exp(lnkr)/r[::-1]

    # transform via real FFT
    cm = np.fft.rfft(ar, axis=-1)
    cm *= um
    ak = np.fft.irfft(cm, n, axis=-1)
    ak[..., :] = ak[..., ::-1]

    # debias output
    ak /= k**(1+q)

    # output grid and transform
    result = (k, ak)

    # derivative
    if deriv:
        cm *= -(1 + q + 1j*y)
        dak = np.fft.irfft(cm, n, axis=-1)
        dak[..., :] = dak[..., ::-1]
        dak /= k**(1+q)
        result = result + (dak,)

    # return chosen outputs
    return result
