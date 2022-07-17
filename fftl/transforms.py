# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
'''
Standard Integral Transforms (:mod:`fftl.transforms`)
=====================================================

The :mod:`fftl.transforms` module provides Python implementations for a number
of standard integral transforms.

.. note::

   The :mod:`fftl.transforms` module requires the ``scipy`` package.

The integral transforms generally accept the same arguments as the :func:`fftl`
routine, except that the coefficient function ``u`` is replaced by the
parameters of the integral transforms.


List of transforms
------------------

.. autosummary::
   :toctree: reference
   :nosignatures:

   hankel
   laplace
   sph_hankel
   stieltjes

'''

import numpy as np
from scipy.special import gamma, loggamma, poch, beta
from . import fftl

PI = np.pi
SRPI = PI**0.5


def cpoch(z, m):
    '''Pochhammer symbol for complex arguments'''
    if np.broadcast(z, m).ndim == 0:
        if np.isreal(z) and np.isreal(m):
            return poch(np.real(z), np.real(m))
        else:
            return np.exp(loggamma(z+m) - loggamma(z))
    return np.where(np.isreal(z) & np.isreal(m),
                    poch(np.real(z), np.real(m)),
                    np.exp(loggamma(z+m) - loggamma(z)))


def cbeta(a, b):
    '''Beta function for complex arguments'''
    if np.broadcast(a, b).ndim == 0:
        if np.isreal(a) and np.isreal(b):
            return beta(np.real(a), np.real(b))
        else:
            return np.exp(loggamma(a) + loggamma(b) - loggamma(a+b))
    return np.where(np.isreal(a) & np.isreal(b),
                    beta(np.real(a), np.real(b)),
                    np.exp(loggamma(a) + loggamma(b) - loggamma(a+b)))


def u_hankel(x, mu):
    '''coefficient function for the Hankel transform'''
    return 2**x*cpoch((1+mu-x)/2, x)


def hankel(mu, r, ar, *args, **kwargs):
    r'''Hankel transform

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

    >>> # some test function
    >>> p, q = 2.0, 0.5
    >>> r = np.logspace(-2, 2, 1000)
    >>> ar = r**p*np.exp(-q*r)
    >>>
    >>> # compute a biased transform
    >>> from fftl.transforms import hankel
    >>> mu = 1.0
    >>> k, ak = hankel(mu, r, ar, q=0.1)

    Compare with the analytical result.

    >>> from scipy.special import gamma, hyp2f1
    >>> res = k*q**(-3-p)*gamma(3+p)*hyp2f1((3+p)/2, (4+p)/2, 2, -(k/q)**2)/2
    >>>
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(k, ak, '-k', label='numerical')
    >>> plt.plot(k, res, ':r', label='analytical')
    >>> plt.xscale('log')
    >>> plt.yscale('symlog', linthresh=1e0, subs=[2, 3, 4, 5, 6, 7, 8, 9])
    >>> plt.ylim(-5e-1, 1e2)
    >>> plt.legend()
    >>> plt.show()

    '''
    return fftl(u_hankel, r, ar*r, *args, args=(mu,), **kwargs)


def u_laplace(x):
    '''coefficient function for the Laplace transform'''
    return gamma(1+x)


def laplace(r, ar, *args, **kwargs):
    r'''Laplace transform

    The Laplace transform is defined as

    .. math::

        \tilde{a}(k) = \int_{0}^{\infty} \! a(r) \, e^{-kr} \, dr \;.

    Examples
    --------
    Compute the Laplace transform.

    >>> # some test function
    >>> p, q = 2.0, 0.5
    >>> r = np.logspace(-2, 2, 1000)
    >>> ar = r**p*np.exp(-q*r)
    >>>
    >>> # compute a biased transform
    >>> from fftl.transforms import laplace
    >>> k, ak = laplace(r, ar, q=0.7)

    Compare with the analytical result.

    >>> from scipy.special import gamma
    >>> res = gamma(p+1)/(q + k)**(p+1)
    >>>
    >>> import matplotlib.pyplot as plt
    >>> plt.loglog(k, ak, '-k', label='numerical')
    >>> plt.loglog(k, res, ':r', label='analytical')
    >>> plt.legend()
    >>> plt.show()

    '''
    return fftl(u_laplace, r, ar, *args, args=(), **kwargs)


def u_sph_hankel(x, mu):
    '''coefficient function for the spherical Hankel transform'''
    return 2**(x-1)*SRPI*cpoch((2+mu-x)/2, (2*x-1)/2)


def sph_hankel(mu, r, ar, *args, **kwargs):
    r'''Hankel transform with spherical Bessel functions

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

    >>> # some test function
    >>> p, q = 2.0, 0.5
    >>> r = np.logspace(-2, 2, 1000)
    >>> ar = r**p*np.exp(-q*r)
    >>>
    >>> # compute a biased transform
    >>> from fftl.transforms import sph_hankel
    >>> mu = 1.0
    >>> k, ak = sph_hankel(mu, r, ar, q=0.1)

    Compare with the analytical result.

    >>> from scipy.special import gamma
    >>> u = (1 + k**2/q**2)**(-p/2)*q**(-p)*gamma(1+p)/(k**2*(k**2 + q**2)**2)
    >>> v = k*(k**2*(2 + p) - p*q**2)*np.cos(p*np.arctan(k/q))
    >>> w = q*(k**2*(3 + 2*p) + q**2)*np.sin(p*np.arctan(k/q))
    >>> res = u*(v + w)
    >>>
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(k, ak, '-k', label='numerical')
    >>> plt.plot(k, res, ':r', label='analytical')
    >>> plt.xscale('log')
    >>> plt.yscale('symlog', linthresh=1e0, subs=[2, 3, 4, 5, 6, 7, 8, 9])
    >>> plt.ylim(-1e0, 1e3)
    >>> plt.legend()
    >>> plt.show()

    '''
    return fftl(u_sph_hankel, r, ar*r**2, *args, args=(mu,), **kwargs)


def u_stieltjes(x, rho):
    '''coefficient function for the Stieltjes transform'''
    return cbeta(1+x, -1-x+rho)


def stieltjes(rho, r, ar, q=0.0, kr=1.0, **kwargs):
    r'''Generalised Stieltjes transform

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

    >>> # some test function
    >>> s = 0.1
    >>> r = np.logspace(-4, 2, 100)
    >>> ar = r/(s + r)**2
    >>>
    >>> # compute a biased transform with shift
    >>> from fftl.transforms import stieltjes
    >>> rho = 2.
    >>> k, ak = stieltjes(rho, r, ar, q=-0.2, kr=1e-2)

    Compare with the analytical result.

    >>> res = (2*(s-k) + (k+s)*np.log(k/s))/(k-s)**3
    >>>
    >>> import matplotlib.pyplot as plt
    >>> plt.loglog(k, ak, '-k', label='numerical')
    >>> plt.loglog(k, res, ':r', label='analytical')
    >>> plt.legend()
    >>> plt.show()

    Compute the derivative in two ways and compare with numerical and
    analytical results.

    >>> # compute Stieltjes transform with derivative
    >>> k, ak, akp = stieltjes(rho, r, ar, kr=1e-1, deriv=True)
    >>>
    >>> # derivative by rho+1 transform
    >>> k_, takp = stieltjes(rho+1, r, ar, kr=1e-1)
    >>> takp *= -rho*k_
    >>>
    >>> # numerical derivative
    >>> nakp = np.gradient(ak, np.log(k))
    >>>
    >>> # analytical derivative
    >>> aakp = -((-5*k**2+4*k*s+s**2+2*k*(k+2*s)*np.log(k/s))/(k-s)**4)
    >>>
    >>> # show
    >>> plt.loglog(k, -akp, '-k', label='deriv')
    >>> plt.loglog(k_, -takp, '-.b', label='rho+1')
    >>> plt.loglog(k, -nakp, ':g', label='numerical')
    >>> plt.loglog(k, -aakp, ':r', label='analytical')
    >>> plt.legend()
    >>> plt.show()

    '''
    kr = r[-1]*r[0]/kr
    k, f, *fp = fftl(u_stieltjes, r, ar, args=(rho,), kr=kr, **kwargs)
    k, f = 1/k[::-1], f[::-1]
    f /= k**rho
    if fp:
        fp[0] = fp[0][::-1]
        fp[0] /= -k**rho
        fp[0] -= rho*f
    return (k, f, *fp)
