# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
"""
:mod:`fftl.transforms` --- Standard Integral Transforms
=======================================================

.. currentmodule:: fftl.transforms

The :mod:`fftl.transforms` module provides Python implementations for a
number of standard integral transforms.

.. autoclass:: HankelTransform
.. autoclass:: LaplaceTransform
.. autoclass:: SphericalHankelTransform
.. autoclass:: StieltjesTransform

"""

from dataclasses import dataclass

import numpy as np
from scipy.special import gamma, loggamma, poch, beta

import fftl

SRPI = np.pi**0.5


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
    >>> from fftl.transforms import HankelTransform
    >>> hankel = HankelTransform(1.0)
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
        fftl.requires(-1.0 + self.mu.real, 0.5, q=q)
        return fftl.transform(self.u, r, ar * r, q=q, **kwargs)


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
    >>> from fftl.transforms import LaplaceTransform
    >>> laplace = LaplaceTransform()
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
        fftl.requires(-1.0, None, q=q)
        return fftl.transform(self.u, r, ar, q=q, **kwargs)


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
    >>> from fftl.transforms import SphericalHankelTransform
    >>> sph_hankel = SphericalHankelTransform(1.0)
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
        fftl.requires(-1.0 + self.mu.real, 1.0, q=q)
        return fftl.transform(self.u, r, ar * r**2, q=q, **kwargs)


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
    >>> from fftl.transforms import StieltjesTransform
    >>> stieltjes = StieltjesTransform(2.)
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
    >>> stieltjes_d = StieltjesTransform(stieltjes.rho+1)
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

        k, ak, *dak = fftl.transform(self.u, r, ar, kr=kr, **kwargs)

        k, ak = 1 / k[::-1], ak[::-1]
        ak /= k**self.rho
        if dak:
            dak[0] = dak[0][::-1]
            dak[0] /= -(k**self.rho)
            dak[0] -= self.rho * ak

        return (k, ak, *dak)
