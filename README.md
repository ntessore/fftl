`fftl` — generalised FFTLog for Python
======================================

The `fftl` package for Python contains a routine to calculate integral
transforms of the type *ã(k) = ∫ a(r) T(kr) dr* for arbitrary kernels *T*.  It
uses a modified FFTLog [2] method of Hamilton [1] to efficiently compute the
transform on logarithmic input and output grids.

Besides the generalised FFTLog algorithm, the package also provides a number of
standard integral transforms.


Installation
------------

Install with pip:

    pip install fftl

For development, it is recommended to clone the GitHub repository, and perform
an editable pip installation.

The core package only requires `numpy`.  The standard integral transform module
additionally requires `scipy`.


Usage
-----

The core functionality of the package is provided by the [`fftl`] module.  The
[`fftl()`] routine computes the generalised FFTLog integral transform for a
given kernel.

For convenience, a number of standard integral transforms are implemented in
the [`fftl.transforms`] module.

[`fftl`]: https://fftl.readthedocs.io/en/latest/fftl.html
[`fftl()`]: https://fftl.readthedocs.io/en/latest/reference/fftl.fftl.html#fftl.fftl
[`fftl.transforms`]: https://fftl.readthedocs.io/en/latest/transforms.html


User manual
-----------

* [Core Functionality (`fftl`)](https://fftl.readthedocs.io/en/latest/fftl.html)
* [Standard Integral Transforms (`fftl.transforms`)](https://fftl.readthedocs.io/en/latest/transforms.html)


References
----------

1.  Hamilton A. J. S., 2000, MNRAS, 312, 257 (astro-ph/9905191)
2.  Talman J. D., 1978, J. Comp. Phys., 29, 35
