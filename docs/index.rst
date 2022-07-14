*****************************************
``fftl`` -- generalised FFTLog for Python
*****************************************

.. currentmodule:: fftl

The ``fftl`` package for Python contains a routine to calculate integral
transforms of the type

.. math::

    \tilde{a}(k) = \int_{0}^{\infty} \! a(r) \, T(kr) \, dr

for arbitrary kernels :math:`T`.  It uses a modified FFTLog [2]_ method of
Hamilton [1]_ to efficiently compute the transform on logarithmic input and
output grids.

The package only requires ``numpy``.  To install with ``pip``::

    pip install fftl


References
==========

.. [1] Hamilton A. J. S., 2000, MNRAS, 312, 257 (astro-ph/9905191)
.. [2] Talman J. D., 1978, J. Comp. Phys., 29, 35


Functions
=========

.. autosummary::
   :toctree: reference
   :nosignatures:

   fftl
