.. _refining disparity:

Refinement step
===============
The purpose of this step is to refine the disparity identified in the previous step.
So, the refinement step involves transforming a pixel disparity map into a sub-pixel disparity map.


Four methods are available in pandora2d:

- Interpolation
- Optical flow
- Dichotomy (c++ version)
- Dichotomy (python version)

.. warning::
    The optical flow method is still in an experimental phase.

Dichotomy
---------

The principle of the dichotomy is to only interpolate values around a specific value instead of the whole dataset.

In the case of the refinement, the dataset is the cost surface and the initial cost value to interpolate around is the one selected at pixelic step.

At each step of the dichotomy, the interpolated values are the 8 values located at half way of the value selected at the previous step and its first neighborhoods.

This means that at first iteration, the distance between initial value and its neighborhoods being of :math:`1` pixel [#]_, the precision is :math:`\frac{1}{2}\times 1 = 0.5` pixel.

At the second iteration, the distance between initial value and its neighborhoods is now :math:`0.5` pixel, thus the precision is :math:`\frac{1}{2}\times0.5=0.25` pixel.

As a matter of fact, at iteration :math:`t`, the reached precision will be :math:`\frac{1}{2^t}` pixel.

At each step, a new best value is selected among the interpolated values using the same method as the one used in :ref:`disparity step<disparity>` (*note that this can still be the initial one if none of the interpolated value is better*).
The position of this new best value will be used in the next step as position to interpolate around.

.. admonition:: Illustration of the points interpolated at each step.
   :name: Dichotomy principle schema

    *Clic on the iteration box in the legend to make corresponding step appear*

    .. container:: html-image

        .. raw:: html
           :file: ../Images/Dichotomy_principle.drawio.html

The method used to do the interpolation depends on the one given in the configuration of the :ref:`refinement step <refinement>`.
It can be one of those listed in :ref:`interpolation_filters` where the fractional shift corresponds to the precision of the iteration step.

.. rubric:: Footnotes
.. [#] In case subpix option was used at :ref:`matching cost step<matching_cost>`, the initial distance could be less than `1`, thus resulting precision will be different.