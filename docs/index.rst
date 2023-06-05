*FFTL* --- Generalised FFTLog for Python
========================================

.. toctree::
   :hidden:

   fftl.rst
   scipy.rst


The *FFTL* package for Python contains a routine to calculate integral
transforms of the type

.. math::

    \tilde{a}(k) = \int_{0}^{\infty} \! a(r) \, T(kr) \, dr

for arbitrary kernels :math:`T`.  It uses a generalisation of the FFTLog [2]_
method of Hamilton [1]_ to efficiently compute the transform on logarithmic
input and output grids.

The package supports any array implementation such as Numpy, JAX, Torch, dask,
etc. as long as it provides a reasonable set of standard Array API functions
(e.g. ``log()``, ``exp()``, ``linspace()``, etc.) and Fast Fourier Transforms
(``fft.rfft()`` and ``fft.irfft()``).

Besides the generalised FFTLog algorithm, the package also provides a number of
standard integral transforms implemented using SciPy.


Installation
------------

Install with pip::

    pip install fftl

For development, it is recommended to clone the `GitHub repository`__, and
perform an editable pip installation.

__ https://github.com/ntessore/fftl

The core package only requires ``numpy``.

The standard integral transforms in the :mod:`fftl.scipy` module additionally
require ``scipy``.  You can install *FFTL* with these additional dependencies
using ``pip install fftl[scipy]``.


Usage
-----

The core functionality of the package is provided by the :mod:`fftl` module.
The :func:`~fftl.fftl` routine computes the generalised FFTLog integral
transform for a given kernel.

For convenience, a number of standard integral transforms are implemented in
the :mod:`fftl.scipy` module.


References
----------

.. [1] Hamilton A. J. S., 2000, MNRAS, 312, 257 (astro-ph/9905191)
.. [2] Talman J. D., 1978, J. Comp. Phys., 29, 35
