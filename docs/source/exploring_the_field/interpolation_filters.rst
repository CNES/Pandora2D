.. _interpolation_filters:

Interpolation filters
=====================

For interpolating data, `Pandora2d` uses its own interpolation filters. Those filters are used in the :ref:`refinement step <refinement>`.

Bicubic
-------

This filter uses the following kernel in 1D:

.. math::
    W(x) =
    \begin{cases}
     (\alpha + 2)|x|^3 - (\alpha + 3) |x|^2 + 1           & \text{if } |x| \leq 1 \\
     \alpha |x|^3 - 5\alpha |x|^2 + 8\alpha |x| - 4\alpha & \text{if } 1 < |x| < 2
    \end{cases}

Where

.. math::
    \alpha = -0.5

and

.. math::
    x = a - \text{fractional\_shift}

In the following image, two lines of an image are sketched with on each a point to interpolate at indices `3.5` and `5.5`:

.. image:: /Images/bicubic_filter_shift.drawio.svg
    :align: center

For both points to interpolate, the `fractional_shift` is `0.5`. Thus, the coefficients to apply to pixels’ values are the same for both points.

.. image:: /Images/bicubic_filter_get_coeff.drawio.svg
    :align: center

:math:`x` values for which kernel is not null are available for :math:`a` ranging from :math:`-1` to :math:`2`.

Thus, for an image in 2D, the filter is applied on an array of shape :math:`4 \times 4` where coefficients are applied on columns then lines.

Sinc
----

This filter use a cardinal sine of the form:

.. math::
    \frac{\sin(x\ \pi)}{x\ \pi}

Where :math:`x` is the fractional shift.

Computed coefficients are windowed by a Gaussian of form:

.. math::
    \exp\left(\frac{-2\ \pi\ x^2}{\sigma^2}\right)

Where :math:`\sigma` correspond to the size of the filter (half width of the window).
