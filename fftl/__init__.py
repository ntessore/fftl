# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
'''
Core Functionality (:mod:`fftl`)
================================

The main functionality of the package is provided by the :func:`fftl` routine
to compute the generalised FFTLog integral transform for a given kernel.


List of functions
-----------------

.. autosummary::
   :toctree: reference
   :nosignatures:

   fftl

'''

__version__ = '2022.7.15'

__all__ = [
    'fftl',
]

from ._core import fftl
