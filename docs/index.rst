``fftl`` -- generalised FFTLog for Python
=========================================

The ``fftl`` package for Python contains a routine to calculate integral
transforms of the type

.. math::

    \tilde{a}(k) = \int_{0}^{\infty} \! a(r) \, T(kr) \, dr

for arbitrary kernels :math:`T`.  It uses a generalisation of the FFTLog [2]_
method of Hamilton [1]_ to efficiently compute the transform on logarithmic
input and output grids.

Besides the generalised FFTLog algorithm, the package also provides a number of
standard integral transforms.


Installation
------------

Install with pip::

    pip install fftl

For development, it is recommended to clone the `GitHub repository`__, and
perform an editable pip installation.

__ https://github.com/ntessore/fftl

The core package only requires ``numpy``.  The standard integral transform
module additionally requires ``scipy``.


Usage
-----

The core functionality of the package is provided by the :mod:`fftl` module.
The :func:`fftl()` routine computes the generalised FFTLog integral transform
for a given kernel.

For convenience, a number of standard integral transforms are implemented in
the :mod:`fftl.transforms` module.


User manual
-----------

.. toctree::
   :maxdepth: 1

   fftl.rst
   transforms.rst


References
----------

.. [1] Hamilton A. J. S., 2000, MNRAS, 312, 257 (astro-ph/9905191)
.. [2] Talman J. D., 1978, J. Comp. Phys., 29, 35
