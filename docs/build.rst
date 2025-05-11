Reusable transforms
===================

The generalised FFTLog transforms depend on the coefficient function *u*, on
the logarithmic grid of input value, and on the parameters *q* and *kr*, but
not on the data to be transformed.  It is therefore possible to pre-compute a
reusable transform that can be applied repeatedly to different input data.

.. autofunction:: fftl.build
