`fftl` — generalised FFTLog for Python
======================================

The `fftl` package for Python contains a routine to calculate integral
transforms of the type *ã(k) = ∫ a(r) T(kr) dr* for arbitrary kernels *T*.  It
uses a modified FFTLog[^2] method of Hamilton[^1] to efficiently compute the
transform on logarithmic input and output grids.

The package only requires `numpy`.  To install with `pip`:

    pip install fftl


[^1]: Hamilton A. J. S., 2000, MNRAS, 312, 257 (astro-ph/9905191)
[^2]: Talman J. D., 1978, J. Comp. Phys., 29, 35
