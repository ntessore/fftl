# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
"""
Generalised FFTLog.
"""

from __future__ import annotations

__all__ = [
    "HankelTransform",
    "LaplaceTransform",
    "SphericalHankelTransform",
    "StieltjesTransform",
    "transform",
]

from dataclasses import dataclass

import numpy as np
from scipy.special import gamma, loggamma, poch, beta

SRPI = np.sqrt(np.pi)


def array_namespace(a):
    """
    Return the Array API namespace for *a*.
    """
    from sys import modules

    if (numpy := modules.get("numpy")) and isinstance(a, numpy.ndarray):
        return numpy
    if (jax := modules.get("jax")) and isinstance(a, jax.Array):
        return jax.numpy
    raise TypeError(f"unknown array type {type(a)!r}")


def requires(_lo: float | None = None, _up: float | None = None, /, **values) -> None:
    """
    Check that *value* lies in the open interval between *lower* and *upper*.
    """
    for name, value in values.items():
        if _lo is not None and not value > _lo:
            raise ValueError(f"expected {name} > {_lo}, got {value}")
        if _up is not None and not value < _up:
            raise ValueError(f"expected {name} < {_up}, got {value}")


def cpoch(z, m):
    """Pochhammer symbol for complex arguments"""
    if np.broadcast(z, m).ndim == 0:
        if np.isreal(z) and np.isreal(m):
            return poch(np.real(z), np.real(m))
        else:
            return np.exp(loggamma(z + m) - loggamma(z))
    return np.where(
        np.isreal(z) & np.isreal(m),
        poch(np.real(z), np.real(m)),
        np.exp(loggamma(z + m) - loggamma(z)),
    )


def cbeta(a, b):
    """Beta function for complex arguments"""
    if np.broadcast(a, b).ndim == 0:
        if np.isreal(a) and np.isreal(b):
            return beta(np.real(a), np.real(b))
        else:
            return np.exp(loggamma(a) + loggamma(b) - loggamma(a + b))
    return np.where(
        np.isreal(a) & np.isreal(b),
        beta(np.real(a), np.real(b)),
        np.exp(loggamma(a) + loggamma(b) - loggamma(a + b)),
    )


@dataclass(frozen=True)
class _Transform:
    """
    Container for transforms.
    """

    k: object
    r: object
    q: object
    um: object

    def __call__(self, ar, *, deriv=False):
        """
        Compute the transform of *ar*.
        """

        # get the Array API namespace
        xp = array_namespace(ar)

        # get pre-computed transform values
        k, r, q, um = self.k, self.r, self.q, self.um

        # input size
        N = r.size

        if ar.shape[-1] != N:
            raise TypeError("last axis of ar must agree with r")

        # bias input
        if q != 0:
            ar = ar * r ** (-q)

        # transform via real FFT
        cm = um * xp.fft.rfft(ar, axis=-1)
        ak = xp.fft.irfft(cm, N, axis=-1)[..., ::-1]

        # debias output
        ak = ak / k ** (1 + q)

        # done if not computing derivative
        if not deriv:
            return ak

        # derivative
        if deriv:
            L = xp.log(r[-1] / r[0])
            y = 2 * xp.pi / L * xp.arange(N // 2 + 1)
            dak = xp.fft.irfft(-(1 + q + 1j * y) * cm, N, axis=-1)[..., ::-1]
            dak = dak / k ** (1 + q)

        # return transform and first derivative
        return ak, dak


def build(u, r, *, q=0.0, kr=1.0, low_ringing=True):
    """
    Build a transform.
    """

    # get the Array API namespace
    xp = array_namespace(r)

    if r.ndim != 1:
        raise TypeError("r must be 1d array")

    # input size
    N = r.size

    # log spacing
    L = xp.log(r[-1] / r[0])

    # make sure given r is logarithmic grid
    dL = L / (N - 1)
    if xp.any(
        xp.abs(xp.log(r[1:] / r[:-1]) - dL) > xp.sqrt(xp.finfo(dL).eps) * xp.abs(dL)
    ):
        raise ValueError("not a logarithmic grid")

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
    # CHANGED: let the RFFT handle this on its own
    # if N % 2 == 0:
    #     um.imag[-1] = 0

    # set up k in log space
    k = xp.exp(lnkr) / r[::-1]

    # store pre-computed transform
    return _Transform(k, r, q, um)


def transform(u, r, ar, *, q=0.0, kr=1.0, low_ringing=True, deriv=False):
    r"""Generalised FFTLog for integral transforms.

    Computes integral transforms for arbitrary kernels using a generalisation
    of Hamilton's method [1]_ for the FFTLog algorithm [2]_.

    The kernel of the integral transform is characterised by the coefficient
    function ``u``, see notes below, which must be callable and accept complex
    input arrays.

    The function to be transformed must be given on a logarithmic grid ``r``.
    The result of the integral transform is similarly computed on a logarithmic
    grid ``k = kr/r``, where ``kr`` is a scalar constant (default: 1) which
    shifts the logarithmic output grid.  The selected value of ``kr`` is
    automatically changed to the nearest low-ringing value if ``low_ringing``
    is true (the default).

    The integral transform can optionally be biased, see notes below.

    The function can optionally at the same time return the derivative of the
    integral transform with respect to the logarithm of ``k``, by setting
    ``deriv`` to true.

    Parameters
    ----------
    u : callable
        Coefficient function.  Must have signature ``u(x)`` and support complex
        input arrays.
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
    >>> k, ak = fftl.transform(u_laplace, r, ar)
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

    >>> k, ak = fftl.transform(u_laplace, r, ar, q=0.5)
    >>>
    >>> plt.loglog(k, ak)                                   # doctest: +SKIP
    >>> plt.loglog(k, lt, ':')                              # doctest: +SKIP
    >>> plt.xlabel('$k$')                                   # doctest: +SKIP
    >>> plt.ylabel('$L[\\tanh](k)$')                        # doctest: +SKIP
    >>> plt.show()

    """
    t = build(u, r, q=q, kr=kr, low_ringing=low_ringing)
    result = t(ar, deriv=deriv)
    if deriv:
        return (t.k, *result)
    return t.k, result


@dataclass
class HankelTransform:
    r"""Hankel transform on a logarithmic grid.

    The Hankel transform is here defined as

    .. math::

        \tilde{a}(k) = \int_{0}^{\infty} \! a(r) \, J_\mu(kr) \, r \, dr \;,

    where :math:`J_\mu` is the Bessel function of order :math:`\mu`.  The order
    can in general be any real or complex number.  The transform is orthogonal
    and normalised: applied twice, the original function is returned.

    The Hankel transform is equivalent to a normalised spherical Hankel
    transform (:func:`sph_hankel`) with the order and bias shifted by one half.
    Special cases are :math:`\mu = 1/2`, which is related to the Fourier sine
    transform,

    .. math::

        \tilde{a}(k)
        = \sqrt{\frac{2}{\pi}} \int_{0}^{\infty} \! a(r) \,
                                    \frac{\sin(kr)}{\sqrt{kr}} \, r \, dr \;,

    and :math:`\mu = -1/2`, which is related to the Fourier cosine transform,

    .. math::

        \tilde{a}(k)
        = \sqrt{\frac{2}{\pi}} \int_{0}^{\infty} \! a(r) \,
                                    \frac{\cos(kr)}{\sqrt{kr}} \, r \, dr \;.

    Examples
    --------
    Compute the Hankel transform for parameter ``mu = 1``.

    >>> import numpy as np
    >>>
    >>> # some test function
    >>> p, q = 2.0, 0.5
    >>> r = np.logspace(-2, 2, 1000)
    >>> ar = r**p*np.exp(-q*r)
    >>>
    >>> # compute a biased transform
    >>> import fftl
    >>> hankel = fftl.HankelTransform(1.0)
    >>> k, ak = hankel(r, ar, q=0.1)

    Compare with the analytical result.

    >>> from scipy.special import gamma, hyp2f1
    >>> res = k*q**(-3-p)*gamma(3+p)*hyp2f1((3+p)/2, (4+p)/2, 2, -(k/q)**2)/2
    >>>
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(k, ak, '-k', label='numerical')            # doctest: +SKIP
    >>> plt.plot(k, res, ':r', label='analytical')          # doctest: +SKIP
    >>> plt.xscale('log')                                   # doctest: +SKIP
    >>> plt.yscale('symlog', linthresh=1e0,
    ...            subs=np.arange(0.1, 1.0, 0.1))           # doctest: +SKIP
    >>> plt.ylim(-5e-1, 1e2)                                # doctest: +SKIP
    >>> plt.legend()                                        # doctest: +SKIP
    >>> plt.show()

    """

    mu: complex

    def u(self, x):
        return 2**x * cpoch((1 + self.mu - x) / 2, x)

    def __call__(self, r, ar, *, q=0.0, **kwargs):
        requires(-1.0 + self.mu.real, 0.5, q=q)
        return transform(self.u, r, ar * r, q=q, **kwargs)


@dataclass
class LaplaceTransform:
    r"""Laplace transform on a logarithmic grid.

    The Laplace transform is defined as

    .. math::

        \tilde{a}(k) = \int_{0}^{\infty} \! a(r) \, e^{-kr} \, dr \;.

    Examples
    --------
    Compute the Laplace transform using JAX.

    >>> import jax.numpy as jnp
    >>>
    >>> # some test function
    >>> p, q = 2.0, 0.5
    >>> r = jnp.logspace(-2, 2, 1000)
    >>> ar = r**p*jnp.exp(-q*r)
    >>>
    >>> # compute a biased transform
    >>> import fftl
    >>> laplace = fftl.LaplaceTransform()
    >>> k, ak = laplace(r, ar, q=0.7)

    Compare with the analytical result.

    >>> from jax.scipy.special import gamma
    >>> res = gamma(p+1)/(q + k)**(p+1)
    >>>
    >>> import matplotlib.pyplot as plt
    >>> plt.loglog(k, ak, '-k', label='numerical')          # doctest: +SKIP
    >>> plt.loglog(k, res, ':r', label='analytical')        # doctest: +SKIP
    >>> plt.legend()                                        # doctest: +SKIP
    >>> plt.show()

    """

    def u(self, x):
        return gamma(1 + x)

    def __call__(self, r, ar, *, q=0.0, **kwargs):
        requires(-1.0, None, q=q)
        return transform(self.u, r, ar, q=q, **kwargs)


@dataclass
class SphericalHankelTransform:
    r"""Hankel transform with spherical Bessel functions.

    The spherical Hankel transform is here defined as

    .. math::

        \tilde{a}(k) = \int_{0}^{\infty} \! a(r) \, j_\mu(kr) \, r^2 \, dr \;,

    where :math:`j_\mu` is the spherical Bessel function of order :math:`\mu`.
    The order can in general be any real or complex number.  The transform is
    orthogonal, but unnormalised: applied twice, the original function is
    multiplied by :math:`\pi/2`.

    The spherical Hankel transform is equivalent to an unnormalised Hankel
    transform (:func:`hankel`) with the order and bias shifted by one half.
    Special cases are :math:`\mu = 0`, which is related to the Fourier sine
    transform,

    .. math::

        \tilde{a}(k)
        = \int_{0}^{\infty} \! a(r) \, \frac{\sin(kr)}{kr} \, r^2 \, dr \;,

    and :math:`\mu = -1`, which is related to the Fourier cosine transform,

    .. math::

        \tilde{a}(k)
        = \int_{0}^{\infty} \! a(r) \, \frac{\cos(kr)}{kr} \, r^2 \, dr \;.

    Examples
    --------
    Compute the spherical Hankel transform for parameter ``mu = 1``.

    >>> import numpy as np
    >>>
    >>> # some test function
    >>> p, q = 2.0, 0.5
    >>> r = np.logspace(-2, 2, 1000)
    >>> ar = r**p*np.exp(-q*r)
    >>>
    >>> # compute a biased transform
    >>> import fftl
    >>> sph_hankel = fftl.SphericalHankelTransform(1.0)
    >>> k, ak = sph_hankel(r, ar, q=0.1)

    Compare with the analytical result.

    >>> from scipy.special import gamma
    >>> u = (1 + k**2/q**2)**(-p/2)*q**(-p)*gamma(1+p)/(k**2*(k**2 + q**2)**2)
    >>> v = k*(k**2*(2 + p) - p*q**2)*np.cos(p*np.arctan(k/q))
    >>> w = q*(k**2*(3 + 2*p) + q**2)*np.sin(p*np.arctan(k/q))
    >>> res = u*(v + w)
    >>>
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(k, ak, '-k', label='numerical')            # doctest: +SKIP
    >>> plt.plot(k, res, ':r', label='analytical')          # doctest: +SKIP
    >>> plt.xscale('log')                                   # doctest: +SKIP
    >>> plt.yscale('symlog', linthresh=1e0,
    ...            subs=np.arange(0.1, 1.0, 0.1))           # doctest: +SKIP
    >>> plt.ylim(-1e0, 1e3)                                 # doctest: +SKIP
    >>> plt.legend()                                        # doctest: +SKIP
    >>> plt.show()

    """

    mu: complex

    def u(self, x):
        return 2 ** (x - 1) * SRPI * cpoch((2 + self.mu - x) / 2, (2 * x - 1) / 2)

    def __call__(self, r, ar, *, q=0.0, **kwargs):
        requires(-1.0 + self.mu.real, 1.0, q=q)
        return transform(self.u, r, ar * r**2, q=q, **kwargs)


@dataclass
class StieltjesTransform:
    r"""Generalised Stieltjes transform on a logarithmic grid.

    The generalised Stieltjes transform is defined as

    .. math::

        \tilde{a}(k) = \int_{0}^{\infty} \! \frac{a(r)}{(k + r)^\rho} \, dr \;,

    where :math:`\rho` is a positive real number.

    The integral can be computed as a :func:`fftl` transform in :math:`k' =
    k^{-1}` if it is rewritten in the form

    .. math::

        \tilde{a}(k) = k^{-\rho} \int_{0}^{\infty} \! a(r) \,
                                        \frac{1}{(1 + k'r)^\rho} \, dr \;.

    Warnings
    --------
    The Stieltjes FFTLog transform is often numerically difficult.

    Examples
    --------
    Compute the generalised Stieltjes transform with ``rho = 2``.

    >>> import numpy as np
    >>>
    >>> # some test function
    >>> s = 0.1
    >>> r = np.logspace(-4, 2, 100)
    >>> ar = r/(s + r)**2
    >>>
    >>> # compute a biased transform with shift
    >>> import fftl
    >>> stieltjes = fftl.StieltjesTransform(2.)
    >>> k, ak = stieltjes(r, ar, kr=1e-2)

    Compare with the analytical result.

    >>> res = (2*(s-k) + (k+s)*np.log(k/s))/(k-s)**3
    >>>
    >>> import matplotlib.pyplot as plt
    >>> plt.loglog(k, ak, '-k', label='numerical')          # doctest: +SKIP
    >>> plt.loglog(k, res, ':r', label='analytical')        # doctest: +SKIP
    >>> plt.legend()                                        # doctest: +SKIP
    >>> plt.show()

    Compute the derivative in two ways and compare with numerical and
    analytical results.

    >>> # compute Stieltjes transform with derivative
    >>> k, ak, akp = stieltjes(r, ar, kr=1e-1, deriv=True)
    >>>
    >>> # derivative by rho+1 transform
    >>> stieltjes_d = fftl.StieltjesTransform(stieltjes.rho+1)
    >>> k_, takp = stieltjes_d(r, ar, kr=1e-1)
    >>> takp *= -stieltjes.rho*k_
    >>>
    >>> # numerical derivative
    >>> nakp = np.gradient(ak, np.log(k))
    >>>
    >>> # analytical derivative
    >>> aakp = -((-5*k**2+4*k*s+s**2+2*k*(k+2*s)*np.log(k/s))/(k-s)**4)
    >>>
    >>> # show
    >>> plt.loglog(k, -akp, '-k', label='deriv')            # doctest: +SKIP
    >>> plt.loglog(k_, -takp, '-.b', label='rho+1')         # doctest: +SKIP
    >>> plt.loglog(k, -nakp, ':g', label='numerical')       # doctest: +SKIP
    >>> plt.loglog(k, -aakp, ':r', label='analytical')      # doctest: +SKIP
    >>> plt.legend()                                        # doctest: +SKIP
    >>> plt.show()

    """

    rho: float

    def u(self, x):
        return cbeta(1 + x, -1 - x + self.rho)

    def __call__(self, r, ar, *, kr=1.0, **kwargs):
        kr = r[-1] * r[0] / kr

        k, ak, *dak = transform(self.u, r, ar, kr=kr, **kwargs)

        k, ak = 1 / k[::-1], ak[::-1]
        ak /= k**self.rho
        if dak:
            dak[0] = dak[0][::-1]
            dak[0] /= -(k**self.rho)
            dak[0] -= self.rho * ak

        return (k, ak, *dak)
