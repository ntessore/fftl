# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
'''
Standard Integral Transforms (:mod:`fftl.transforms`)
=====================================================

The :mod:`fftl.transforms` module provides implementations for a number of
standard integral transforms.

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

   sph_hankel

'''

import numpy as np
from scipy.special import loggamma
from . import fftl

PI = np.pi
LNPI = np.log(PI)
LN2 = np.log(2)


def u_sph_hankel(x, mu):
    '''coefficient function for the spherical Hankel transform'''

    if not np.all((np.real(x) < 1) & (np.real(x + mu) > -1)):
        raise ValueError('spherical Hankel transform is ill-defined')

    return np.exp(LNPI/2 - (1 - x)*LN2
                  + loggamma((1 + mu + x)/2)
                  - loggamma((2 + mu - x)/2))


def sph_hankel(mu, r, ar, *args, **kwargs):
    r'''Hankel transform with spherical Bessel functions

    The spherical Hankel transform is here defined as

    .. math::

        \tilde{a}(k) = \int_{0}^{\infty} \! a(r) \, j_\mu(kr) \, r^2 \, dr \;,

    where :math:`j_\mu` is the spherical Bessel function of order :math:`\mu`.
    The order can in general be any real or complex number.  The transform is
    orthogonal, but unnormalised: applied twice, the original function is
    multiplied by :math:`\pi/2`.

    Common special cases are :math:`\mu = 0`, which is related to the Fourier
    sine transform,

    .. math::

        \tilde{a}(k)
        = \int_{0}^{\infty} \! a(r) \, \frac{\sin(kr)}{kr} \, r^2 \, dr \;,

    and :math:`\mu = -1`, which is related to the Fourier cosine transform,

    .. math::

        \tilde{a}(k)
        = \int_{0}^{\infty} \! a(r) \, \frac{\cos(kr)}{kr} \, r^2 \, dr \;.

    Internally, the transform is computed as

    .. math::

        \tilde{a}(k)
        = k^{\frac{\mu}{2}} \int_{0}^{\infty} \!
                \bigl[a(r) \, r^{2+\frac{\mu}{2}}\bigr] \,
                    \bigl[(kr)^{-\frac{\mu}{2}} \, j_\mu(kr)\bigr] \, dr \;,

    which is well-defined if :math:`|q| < 1 + \mathrm{Re} \, \frac{\mu}{2}`,
    where :math:`q` is the bias parameter of the :func:`fftl` transform.

    Examples
    --------
    Compute the spherical Hankel transform for parameter ``mu = 1``.

    >>> # some test function
    >>> x = 1/25
    >>> r = np.logspace(-4, 2, 100)
    >>> ar = 1/(r + x)**4
    >>>
    >>> # compute a biased transform
    >>> from fftl.transforms import sph_hankel
    >>> mu = 1.0
    >>> k, ak = sph_hankel(mu, r, ar, q=0.22)

    Compare with the analytical result.

    >>> from scipy.special import sici
    >>> si, ci = sici(k*x)
    >>> u = np.pi*k*x*np.cos(k*x) + 2*np.pi*np.sin(k*x) - 2
    >>> v = k*x*np.sin(k*x) - 2*np.cos(k*x)
    >>> w = k*x*np.cos(k*x) + 2*np.sin(k*x)
    >>> res = k*u/12 + k*ci*v/6 - k*si*w/6
    >>>
    >>> import matplotlib.pyplot as plt
    >>> plt.loglog(k, ak, '-k', label='numerical')
    >>> plt.loglog(k, res, ':r', label='analytical')
    >>> plt.legend()
    >>> plt.show()

    '''
    if len(args) > 0:
        q, *args = args
    else:
        q = kwargs.pop('q', 0.) - mu/2
    return fftl(u_sph_hankel, r, ar*r**2, q, *args, args=(mu,), **kwargs)
