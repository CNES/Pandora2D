.. _refining disparity:

Refinement step
===============
The purpose of this step is to refine the disparity identified in the previous step.
So, the refinement step involves transforming a pixel disparity map into a sub-pixel disparity map.


Two methods are available in pandora2d:

- Interpolation
- Optical flow.

.. warning::
    The optical flow method is still in an experimental phase.
