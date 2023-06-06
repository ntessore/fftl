# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT

__version__ = '2023.6'

__all__ = [
    'newfftl',
]


def _wrap(fftl):

    def decorator(func):
        if func.__kwdefaults__ is None:
            kwdefaults = func.__kwdefaults__ = {}
        else:
            kwdefaults = func.__kwdefaults__
        for key, value in fftl.__kwdefaults__.items():
            kwdefaults.setdefault(key, value)
        if not func.__doc__:
            func.__doc__ = fftl.__doc__
        return func

    return decorator


def newfftl(xp):

    def fftl(u, r, ar, *, q=0.0, kr=1.0, low_ringing=True, deriv=False):
        r'''Generalised FFTLog for integral transforms.'''

        if r.ndim != 1:
            raise TypeError('r must be 1D array')
        if ar.shape[-1] != r.shape[-1]:
            raise TypeError('last axis of ar must agree with r')

        # inputs
        n, = r.shape

        # log spacing
        dlnr = xp.log(r[-1]/r[0])/(n-1)

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

        # fix last coefficient to real when n is even
        if not n & 1:
            um = um - 1j*(y == y[-1])*um.imag

        # bias input
        ar = ar * r**(-q)

        # set up k in log space
        k = xp.exp(lnkr)/r[::-1]

        # transform via real FFT
        cm = xp.fft.rfft(ar, axis=-1)
        cm *= um
        ak = xp.fft.irfft(cm, n, axis=-1)

        # reorder and debias output
        ak = ak[..., ::-1] * k**(-1-q)

        # output grid and transform
        result = (k, ak)

        # derivative
        if deriv:
            cm *= -(1 + q + 1j*y)
            dak = xp.fft.irfft(cm, n, axis=-1)
            dak = dak[..., ::-1] * k**(-1-q)
            result = result + (dak,)

        # return chosen outputs
        return result

    # decorator for easy wrapping
    fftl.wrap = _wrap(fftl)

    return fftl
