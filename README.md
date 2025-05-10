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
standard integral transforms.


Installation
------------

Install with pip:

    pip install fftl

The package only requires `numpy` and `scipy`.

For development, it is recommended to clone the GitHub repository, and perform
an editable pip installation.


Usage
-----

The core functionality of the package is provided by the [`fftl`] module.  The
[`fftl.transform()`] routine computes the generalised FFTLog integral transform
for a given kernel.  For convenience, the module provides a number of
[standard integral transforms].

[`fftl`]: https://fftl.readthedocs.io/latest/fftl.html
[`fftl.transform()`]: https://fftl.readthedocs.io/latest/fftl.html#fftl.transform
[standard integral transforms]: https://fftl.readthedocs.io/latest/transforms.html


User manual
-----------

* [Generalised FFTLog][`fftl`]
* [Standard Integral Transforms][standard integral transforms]


References
----------

1.  Hamilton A. J. S., 2000, MNRAS, 312, 257 (astro-ph/9905191)
2.  Talman J. D., 1978, J. Comp. Phys., 29, 35
