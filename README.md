*FFTL* — generalised FFTLog for Python
======================================

The *FFTL* package for Python contains a routine to calculate integral
transforms of the type *ã(k) = ∫ a(r) T(kr) dr* for arbitrary kernels *T*.  It
uses a modified FFTLog [2] method of Hamilton [1] to efficiently compute the
transform on logarithmic input and output grids.

The package supports any array implementation such as Numpy, JAX, Torch, dask,
etc. as long as it provides a reasonable set of standard Array API functions
(e.g. `log()`, `exp()`, `linspace()`, etc.) and Fast Fourier Transforms
(`fft.rfft()` and `fft.irfft()`).

Besides the generalised FFTLog algorithm, the package also provides a number of
standard integral transforms implemented using SciPy.


Installation
------------

Install with pip:

    pip install fftl

For development, it is recommended to clone the GitHub repository, and perform
an editable pip installation.

The core package only requires `numpy`.

The standard integral transforms in the `fftl.scipy` module additionally
require `scipy`.  You can install *FFTL* with these additional dependencies
using `pip install fftl[scipy]`.


Usage
-----

The core functionality of the package is provided by the [`fftl`] module.  The
[`fftl()`] routine computes the generalised FFTLog integral transform for a
given kernel.

For convenience, a number of standard integral transforms are implemented in
the [`fftl.scipy`] module.

[`fftl`]: https://fftl.readthedocs.io/en/latest/fftl.html
[`fftl()`]: https://fftl.readthedocs.io/en/latest/fftl.html#fftl.fftl
[`fftl.scipy`]: https://fftl.readthedocs.io/en/latest/scipy.html


User manual
-----------

* [`fftl` — Generalised FFTLog][`fftl`]
* [`fftl.scipy` — Standard Integral Transforms using SciPy][`fftl.scipy`]


References
----------

1.  Hamilton A. J. S., 2000, MNRAS, 312, 257 (astro-ph/9905191)
2.  Talman J. D., 1978, J. Comp. Phys., 29, 35
