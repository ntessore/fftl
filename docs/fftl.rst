.. module:: fftl

:mod:`fftl` --- Generalised FFTLog
==================================

The functionality of the :mod:`fftl` module is provided by the
:func:`fftl` routine to compute the generalised FFTLog integral
transform for a given kernel.


Creating new interfaces
-----------------------

.. autofunction:: newfftl


Common interface
----------------

.. np:function:: fftl

   Computes integral transforms for arbitrary kernels using a generalisation
   of Hamilton's method [1]_ for the FFTLog algorithm [2]_.

   The kernel of the integral transform is characterised by the coefficient
   function ``u``, see notes below, which must be callable and accept complex
   input arrays.  Additional arguments for ``u`` can be passed with the
   optional ``args`` parameter.

   The function to be transformed must be given on a logarithmic grid ``r``.
   The result of the integral transform is similarly computed on a logarithmic
   grid ``k = kr/r``, where ``kr`` is a scalar constant (default: 1) which
   shifts the logarithmic output grid.  The selected value of ``kr`` is
   automatically changed to the nearest low-ringing value if ``krgood`` is
   true (the default).

   The integral transform can optionally be biased, see notes below.

   The function can optionally at the same time return the derivative of the
   integral transform with respect to the logarithm of ``k``, by setting
   ``deriv`` to true.

   Parameters
   ----------
   u : callable
       Coefficient function.  Must have signature ``u(x)`` and support
       complex-valued array input.
   r : array_like (N,)
       Grid of input points.  Must have logarithmic spacing.
   ar : array_like (..., N)
       Function values.  If multidimensional, the integral transform applies
       to the last axis, which must agree with input grid.
   q : float, optional
       Bias parameter for integral transform.
   kr : float, optional
       Shift parameter for logarithmic output grid.
   low_ringing : bool, optional
       Change given ``kr`` to the nearest value fulfilling the low-ringing
       condition.
   deriv : bool, optional
       Also return the first derivative of the integral transform.

   Returns
   -------
   k : array_like (N,)
       Grid of output points.
   ak : array_like (..., N)
       Integral transform evaluated at ``k``.
   dak : array_like (..., N), optional
       If ``deriv`` is true, the derivative of ``ak`` with respect to the
       logarithm of ``k``.

   Other Parameters
   ----------------
   xp : str or object
       An optional namespace that contains array functions such as
       ``xp.log()``, ``xp.exp()``, etc.  If a string, a module with
       that name will be imported.

   Notes
   -----
   Computes integral transforms of the form

   .. math::

       \tilde{a}(k) = \int_{0}^{\infty} \! a(r) \, T(kr) \, dr

   for arbitrary kernels :math:`T`.

   If :math:`a(r)` is given on a logarithmic grid of :math:`r` values, the
   integral transform can be computed for a logarithmic grid of :math:`k`
   values with a modification of Hamilton's FFTLog algorithm,

   .. math::

       U(x) = \int_{0}^{\infty} \! t^x \, T(t) \, dt \;.

   The generalised FFTLog algorithm therefore only requires the coefficient
   function :math:`U` for the given kernel.  Everything else, and in
   particular how to construct a well-defined transform, remains exactly the
   same as in Hamilton's original algorithm.

   The transform can optionally be biased,

   .. math::

       \tilde{a}(k) = k^{-q} \int_{0}^{\infty} \! [a(r) \, r^{-q}] \,
                                                   [T(kr) \, (kr)^q] \, dr \;,

   where :math:`q` is the bias parameter.  The respective biasing factors
   :math:`r^{-q}` and :math:`k^{-q}` for the input and output values are
   applied internally.

   References
   ----------
   .. [1] Hamilton A. J. S., 2000, MNRAS, 312, 257 (astro-ph/9905191)
   .. [2] Talman J. D., 1978, J. Comp. Phys., 29, 35

   Examples
   --------
   Compute the one-sided Laplace transform of the hyperbolic tangent function.
   The kernel of the Laplace transform is :math:`\exp(-kt)`, which determines
   the coefficient function.

   >>> import numpy as np
   >>> from scipy.special import gamma, digamma
   >>>
   >>> def u_laplace(x):
   ...     # requires Re(x) = q > -1
   ...     return gamma(1 + x)

   Create the input function values on a logarithmic grid.

   >>> r = np.logspace(-4, 4, 100)
   >>> ar = np.tanh(r)
   >>>
   >>> import matplotlib.pyplot as plt
   >>> plt.loglog(r, ar)                                   # doctest: +SKIP
   >>> plt.xlabel('$r$')                                   # doctest: +SKIP
   >>> plt.ylabel('$\\tanh(r)$')                           # doctest: +SKIP
   >>> plt.show()

   Compute the Laplace transform, and compare with the analytical result.

   >>> from fftl import newfftl
   >>>
   >>> fftl = newfftl(np)
   >>> k, ak = fftl(u_laplace, r, ar)
   >>>
   >>> lt = (digamma((k+2)/4) - digamma(k/4) - 2/k)/2
   >>>
   >>> plt.loglog(k, ak)                                   # doctest: +SKIP
   >>> plt.loglog(k, lt, ':')                              # doctest: +SKIP
   >>> plt.xlabel('$k$')                                   # doctest: +SKIP
   >>> plt.ylabel('$L[\\tanh](k)$')                        # doctest: +SKIP
   >>> plt.show()

   The numerical Laplace transform has an issue on the right, which is due to
   the circular nature of the FFTLog integral transform.  The effect is
   mitigated by computing a biased transform with ``q = 0.5``.  Good values of
   the bias parameter ``q`` depend on the shape of the input function.

   >>> k, ak = fftl(u_laplace, r, ar, q=0.5)
   >>>
   >>> plt.loglog(k, ak)                                   # doctest: +SKIP
   >>> plt.loglog(k, lt, ':')                              # doctest: +SKIP
   >>> plt.xlabel('$k$')                                   # doctest: +SKIP
   >>> plt.ylabel('$L[\\tanh](k)$')                        # doctest: +SKIP
   >>> plt.show()


.. decorator:: fftl.wrap

   Decorator for easy wrapping of :func:`fftl`.
