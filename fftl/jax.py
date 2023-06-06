# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
'''
:mod:`fftl.jax` --- FFTL for JAX
================================

.. currentmodule:: fftl.jax

The :mod:`fftl.jax` module provides JAX implementations for a number of
standard integral transforms.

Interface
---------

.. autofunction:: fftl

Standard Integral Transforms
----------------------------

.. autoclass:: HankelTransform
.. autoclass:: LaplaceTransform
.. autoclass:: SphericalHankelTransform
.. autoclass:: StieltjesTransform

'''

from dataclasses import dataclass
import jax.numpy as jnp
from jax import custom_jvp, jit, lax
from . import newfftl

SR_PI = jnp.pi**0.5
LN_PI = jnp.log(jnp.pi)
LN_SR_2PI = jnp.log(2*jnp.pi)/2


def digamma(z):
    '''Compute digamma(z) following Kölbig (1972).'''

    X0 = 7.

    # these are C[k] = b[k]/b[k-1], k = 1, 2, 3, ...,
    # where b[k] = B[2k]/2k, k > 0, with B[2k] the Bernoulli coefficient
    # and b[0] was set to unity
    DIGAM_C = (
        0.083333333333333333333333333333333333333333333333333,
        -0.10000000000000000000000000000000000000000000000000,
        -0.47619047619047619047619047619047619047619047619048,
        -1.0500000000000000000000000000000000000000000000000,
        -1.8181818181818181818181818181818181818181818181818,
        -2.7842490842490842490842490842490842490842490842491,
        -3.9507959479015918958031837916063675832127351664255,
        -5.3191176470588235294117647058823529411764705882353,
        -6.8897614970981966020017327439878841883992997884999,
        -8.6629363965872047117630357879292157907062049127839,
        -10.638716670322826914883842419410301607075404632359,
        -12.817127438511138934655019255253540747149387392426,
        # -15.198176866891341798615255817348329785001055849892,
        # -17.781867530496201807977033961060480139438162444348,
        # -20.568200221926880643403873557351656580971638924982,
        # -23.557175180388933362770162012497160914155672721662,
        # -26.748792476906204901749058388489603740929195010834,
        # -30.143052132279567979802139252370977832315962353185,
        # -33.739954152530241976514641568519088474590403161705,
        # -37.539498539383718370168841295418657387101825078851,
    )

    # map left half-plane to right half-plane, keep track of correction
    # to improve accuracy and handle the case of negative integers correctly,
    # use the fact that tan(pi*z) has unit period in x

    ncor = jnp.where(z.real < 0,
                     1/z + jnp.pi/jnp.tan(jnp.pi*(z - jnp.ceil(z.real))),
                     0)
    z = jnp.where(z.real < 0, -z, z)

    # range reduction: map z to x >= X0, keep track of correction

    rcor = jnp.zeros_like(z)
    for _ in range(int(X0)):
        u = (z.real < X0)
        rcor = rcor + u/z
        z = z + u

    # compute digamma for x >= X0

    r = 1/(z*z)
    s = jnp.zeros_like(z)
    t = jnp.ones_like(z)
    for c in DIGAM_C:
        t = c*r*t
        s = s + t

    log_z = jnp.log(z)

    return log_z - 1/(2*z) - s - rcor - ncor


def _log_sin_pi(z):
    '''Compute ln sin(pi*z).'''
    x = z.real
    pi_xi = jnp.pi * (x - jnp.floor(x))
    if jnp.iscomplexobj(z):
        conj = (z.imag < 0)
        pi_y = jnp.pi * jnp.where(conj, -z.imag, z.imag)
        u = pi_y + jnp.log(jnp.exp(-2*pi_y) * jnp.sin(pi_xi)**2
                           + jnp.expm1(-2*pi_y)**2/4)/2
        v = jnp.arctan(jnp.tanh(pi_y)/jnp.tan(pi_xi)) - jnp.floor(x)*jnp.pi
        v = jnp.where(conj, -v, v)
    else:
        u = jnp.log(jnp.abs(jnp.sin(pi_xi)))
        v = -jnp.pi*jnp.floor(x)
    return lax.complex(u, v)


@custom_jvp
def loggamma(z):
    '''Compute ln Gamma(z) following Kölbig (1972).'''

    X0 = 7.

    # these are C[k] = b[k]/b[k-1], k = 1, 2, 3, ...,
    # where b[k] = B[2k]/2k/(2k-1), k > 0, with B[2k] the Bernoulli coefficient
    # and b[0] was set to unity
    GAMMA_C = (
        0.083333333333333333333333333333333333333333333333333,
        -0.033333333333333333333333333333333333333333333333333,
        -0.28571428571428571428571428571428571428571428571429,
        -0.75000000000000000000000000000000000000000000000000,
        -1.4141414141414141414141414141414141414141414141414,
        -2.2780219780219780219780219780219780219780219780220,
        -3.3429811866859623733719247467438494934876989869754,
        -4.6099019607843137254901960784313725490196078431373,
        -6.0792013209689970017662347741069566368229115780882,
        -7.7510483548411831631564004418314036022108149219645,
        -9.6255055588635100658472859985140824064015565721345,
        -11.702594617771039896858930624361928508266831966998,
        # -13.982322717540034454726035351960463402200971381901,
        # -16.464692157866853525904661075056000129109409670693,
        # -19.149703654897440599031192622361887161594284516363,
        # -22.037357426815453790978538656852182790661758352522,
        # -25.127653538911889453158206364944779271781971070784,
        # -28.420592010435021238099159866521207670469335933003,
        # -31.916172846988066734540877159409948557044975963775,
        # -35.614396050184553325544798152063854444173526356859,
    )

    # map left half-plane to right half-plane, keep track of correction

    ncor = jnp.where(z.real < 0, _log_sin_pi(z) - LN_PI, 0)
    nfac = jnp.where(z.real < 0, -1, +1)
    z = jnp.where(z.real < 0, 1 - z, z)

    # range reduction: map z to x >= X0, keep track of correction

    rcor = jnp.zeros_like(z)
    if jnp.iscomplexobj(z):
        for _ in range(int(X0)):
            u = (z.real < X0)
            rcor = rcor + u*lax.complex(jnp.log(jnp.abs(z)), jnp.angle(z))
            z = z + u
    else:
        for _ in range(int(X0)):
            u = (z < X0)
            rcor = rcor + u*jnp.log(z)
            z = z + u

    # compute loggamma for x >= X0

    r = 1/(z*z)
    s = jnp.zeros_like(z)
    t = jnp.ones_like(z)
    for c in GAMMA_C:
        t = c*r*t
        s = s + t

    log_z = jnp.log(z)

    return nfac*((z - 0.5)*log_z - z + z*s + LN_SR_2PI - rcor) - ncor


@loggamma.defjvp
def loggamma_jvp(primals, tangents):
    z, = primals
    t, = tangents
    return loggamma(z), digamma(z) * t


def poch(z, m):
    '''Pochhammer symbol for complex arguments'''
    return jnp.exp(loggamma(z+m) - loggamma(z))


def beta(a, b):
    '''Beta function for complex arguments'''
    return jnp.exp(loggamma(a) + loggamma(b) - loggamma(a+b))


fftl = jit(newfftl(jnp), static_argnames=('u', 'deriv'))


@dataclass(frozen=True)
class HankelTransform:
    r'''Hankel transform on a logarithmic grid.

    The Hankel transform is here defined as

    .. math::

        \tilde{a}(k) = \int_{0}^{\infty} \! a(r) \, J_\mu(kr) \, r \, dr \;,

    where :math:`J_\mu` is the Bessel function of order :math:`\mu`.  The order
    can in general be any real or complex number.  The transform is orthogonal
    and normalised: applied twice, the original function is returned.

    The Hankel transform is equivalent to a normalised spherical Hankel
    transform  with the order and bias shifted by one half.  Special cases are
    :math:`\mu = 1/2`, which is related to the Fourier sine transform,

    .. math::

        \tilde{a}(k)
        = \sqrt{\frac{2}{\pi}} \int_{0}^{\infty} \! a(r) \,
                                    \frac{\sin(kr)}{\sqrt{kr}} \, r \, dr \;,

    and :math:`\mu = -1/2`, which is related to the Fourier cosine transform,

    .. math::

        \tilde{a}(k)
        = \sqrt{\frac{2}{\pi}} \int_{0}^{\infty} \! a(r) \,
                                    \frac{\cos(kr)}{\sqrt{kr}} \, r \, dr \;.

    See Also
    --------
    SphericalHankelTransform :
        Equivalent transform with order and bias shifted by one half.

    Examples
    --------
    Compute the Hankel transform for parameter ``mu = 1``.

    >>> import jax.numpy as jnp
    >>>
    >>> # some test function
    >>> p, q = 2.0, 0.5
    >>> r = jnp.logspace(-2, 2, 1000)
    >>> ar = r**p*jnp.exp(-q*r)
    >>>
    >>> # compute a biased transform
    >>> from fftl.jax import HankelTransform
    >>> hankel = HankelTransform(1.0)
    >>> k, ak = hankel.fftl(r, ar, q=0.1)

    Compare with the analytical result.

    >>> from scipy.special import gamma, hyp2f1
    >>> res = k*q**(-3-p)*gamma(3+p)*hyp2f1((3+p)/2, (4+p)/2, 2, -(k/q)**2)/2
    >>>
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(k, ak, '-k', label='numerical')            # doctest: +SKIP
    >>> plt.plot(k, res, ':r', label='analytical')          # doctest: +SKIP
    >>> plt.xscale('log')                                   # doctest: +SKIP
    >>> plt.yscale('symlog', linthresh=1e0,
    ...            subs=[2, 3, 4, 5, 6, 7, 8, 9])           # doctest: +SKIP
    >>> plt.ylim(-5e-1, 1e2)                                # doctest: +SKIP
    >>> plt.legend()                                        # doctest: +SKIP
    >>> plt.show()

    '''

    mu: complex

    def __call__(self, x):
        return 2**x*poch((1+self.mu-x)/2, x)

    @fftl.wrap
    def fftl(self, r, ar, **kwargs):
        return fftl(self, r, ar*r, **kwargs)


@dataclass(frozen=True)
class LaplaceTransform:
    r'''Laplace transform on a logarithmic grid.

    The Laplace transform is defined as

    .. math::

        \tilde{a}(k) = \int_{0}^{\infty} \! a(r) \, e^{-kr} \, dr \;.

    Examples
    --------
    Compute the Laplace transform.

    >>> import jax.numpy as jnp
    >>>
    >>> # some test function
    >>> p, q = 2.0, 0.5
    >>> r = jnp.logspace(-2, 2, 1000)
    >>> ar = r**p*jnp.exp(-q*r)
    >>>
    >>> # compute a biased transform
    >>> from fftl.jax import LaplaceTransform
    >>> laplace = LaplaceTransform()
    >>> k, ak = laplace.fftl(r, ar, q=0.7)

    Compare with the analytical result.

    >>> from scipy.special import gamma
    >>> res = gamma(p+1)/(q + k)**(p+1)
    >>>
    >>> import matplotlib.pyplot as plt
    >>> plt.loglog(k, ak, '-k', label='numerical')          # doctest: +SKIP
    >>> plt.loglog(k, res, ':r', label='analytical')        # doctest: +SKIP
    >>> plt.legend()                                        # doctest: +SKIP
    >>> plt.show()

    '''

    def __call__(self, x):
        return jnp.exp(loggamma(1+x))

    fftl = fftl


@dataclass(frozen=True)
class SphericalHankelTransform:
    r'''Hankel transform with spherical Bessel functions.

    The spherical Hankel transform is here defined as

    .. math::

        \tilde{a}(k) = \int_{0}^{\infty} \! a(r) \, j_\mu(kr) \, r^2 \, dr \;,

    where :math:`j_\mu` is the spherical Bessel function of order :math:`\mu`.
    The order can in general be any real or complex number.  The transform is
    orthogonal, but unnormalised: applied twice, the original function is
    multiplied by :math:`\pi/2`.

    The spherical Hankel transform is equivalent to an unnormalised Hankel
    transform with the order and bias shifted by one half.  Special cases are
    :math:`\mu = 0`, which is related to the Fourier sine transform,

    .. math::

        \tilde{a}(k)
        = \int_{0}^{\infty} \! a(r) \, \frac{\sin(kr)}{kr} \, r^2 \, dr \;,

    and :math:`\mu = -1`, which is related to the Fourier cosine transform,

    .. math::

        \tilde{a}(k)
        = \int_{0}^{\infty} \! a(r) \, \frac{\cos(kr)}{kr} \, r^2 \, dr \;.

    See Also
    --------
    HankelTransform :
        Equivalent transform with order and bias shifted by one half.

    Examples
    --------
    Compute the spherical Hankel transform for parameter ``mu = 1``.

    >>> import jax.numpy as jnp
    >>>
    >>> # some test function
    >>> p, q = 2.0, 0.5
    >>> r = jnp.logspace(-2, 2, 1000)
    >>> ar = r**p*jnp.exp(-q*r)
    >>>
    >>> # compute a biased transform
    >>> from fftl.jax import SphericalHankelTransform
    >>> sph_hankel = SphericalHankelTransform(1.0)
    >>> k, ak = sph_hankel.fftl(r, ar, q=0.1)

    Compare with the analytical result.

    >>> from scipy.special import gamma
    >>> u = (1 + k**2/q**2)**(-p/2)*q**(-p)*gamma(1+p)/(k**2*(k**2 + q**2)**2)
    >>> v = k*(k**2*(2 + p) - p*q**2)*jnp.cos(p*jnp.arctan(k/q))
    >>> w = q*(k**2*(3 + 2*p) + q**2)*jnp.sin(p*jnp.arctan(k/q))
    >>> res = u*(v + w)
    >>>
    >>> import matplotlib.pyplot as plt
    >>> plt.plot(k, ak, '-k', label='numerical')            # doctest: +SKIP
    >>> plt.plot(k, res, ':r', label='analytical')          # doctest: +SKIP
    >>> plt.xscale('log')                                   # doctest: +SKIP
    >>> plt.yscale('symlog', linthresh=1e0,
    ...            subs=[2, 3, 4, 5, 6, 7, 8, 9])           # doctest: +SKIP
    >>> plt.ylim(-1e0, 1e3)                                 # doctest: +SKIP
    >>> plt.legend()                                        # doctest: +SKIP
    >>> plt.show()

    '''

    mu: complex

    def __call__(self, x):
        return 2**(x-1)*SR_PI*poch((2+self.mu-x)/2, (2*x-1)/2)

    @fftl.wrap
    def fftl(self, r, ar, **kwargs):
        return fftl(self, r, ar*r**2, **kwargs)


@dataclass(frozen=True)
class StieltjesTransform:
    r'''Generalised Stieltjes transform on a logarithmic grid.

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

    >>> import jax.numpy as jnp
    >>>
    >>> # some test function
    >>> s = 0.1
    >>> r = jnp.logspace(-4, 2, 100)
    >>> ar = r/(s + r)**2
    >>>
    >>> # compute a biased transform with shift
    >>> from fftl.jax import StieltjesTransform
    >>> stieltjes = StieltjesTransform(2.)
    >>> k, ak = stieltjes.fftl(r, ar, kr=1e-2)

    Compare with the analytical result.

    >>> res = (2*(s-k) + (k+s)*jnp.log(k/s))/(k-s)**3
    >>>
    >>> import matplotlib.pyplot as plt
    >>> plt.loglog(k, ak, '-k', label='numerical')          # doctest: +SKIP
    >>> plt.loglog(k, res, ':r', label='analytical')        # doctest: +SKIP
    >>> plt.legend()                                        # doctest: +SKIP
    >>> plt.show()

    Compute the derivative in two ways and compare with numerical and
    analytical results.

    >>> # compute Stieltjes transform with derivative
    >>> k, ak, akp = stieltjes.fftl(r, ar, kr=1e-1, deriv=True)
    >>>
    >>> # derivative by rho+1 transform
    >>> stieltjes_d = StieltjesTransform(stieltjes.rho+1)
    >>> k_, takp = stieltjes_d.fftl(r, ar, kr=1e-1)
    >>> takp *= -stieltjes.rho*k_
    >>>
    >>> # analytical derivative
    >>> aakp = -((-5*k**2+4*k*s+s**2+2*k*(k+2*s)*jnp.log(k/s))/(k-s)**4)
    >>>
    >>> # show
    >>> plt.loglog(k, -akp, '-k', label='deriv')            # doctest: +SKIP
    >>> plt.loglog(k_, -takp, '-.b', label='rho+1')         # doctest: +SKIP
    >>> plt.loglog(k, -aakp, ':r', label='analytical')      # doctest: +SKIP
    >>> plt.legend()                                        # doctest: +SKIP
    >>> plt.show()

    '''

    rho: float

    def __call__(self, x):
        return beta(1+x, -1-x+self.rho)

    @fftl.wrap
    def fftl(self, r, ar, *, kr, **kwargs):
        kr = r[-1]*r[0]/kr

        k, ak, *dak = fftl(self, r, ar, kr=kr, **kwargs)

        k, ak = 1/k[::-1], ak[::-1]
        ak = ak * k**(-self.rho)
        if dak:
            dak[0] = -dak[0][::-1] * k**(-self.rho) - self.rho*ak

        return (k, ak, *dak)
