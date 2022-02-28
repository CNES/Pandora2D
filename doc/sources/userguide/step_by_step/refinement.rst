.. _refinement:

Refinement of the disparity maps
================================

Theoretical basics
------------------
The purpose of this step is to refine the disparity identified in the previous step.

The available refinement method is :

* **Interpolation** : It consists on 3 differents steps:

    * First, the cost_volumes is reshaped to obtain the 2D (disp_row, disp_col) costs map for each pixel, so we will obtain (row * col) 2D cost maps.
    * The cost map of each pixel is interpolated using scipy to obtain a continuous function.
    * Then, the interpolated functions are minimized using scipy to obtain the refined disparities.

* **Optical flow** : Inspired by Lucas & Kanade's algorithm

    * We first need to suppose that pixel's shifting are subpixel between left and right images.
    * Second, we need to suppose brightness constancy between left and right images. (2)
    * Now, we can write :

    .. math::

        I(x, y, t) &= I(x + dx, y + dy, t + dt) \\
        I(x, y, t) &=  I(x, y, t) + \frac{\partial I}{\partial x}\partial x + \frac{\partial I}{\partial y}\partial y +\frac{\partial I}{\partial t}\partial t

    with hypothesis (2) :

    .. math::

         \frac{\partial I}{\partial x} dx + \frac{\partial I}{\partial y} dy + \frac{\partial I}{\partial t}dt = 0

    after dividing by :math:`dt`:

    .. math::

         \frac{\partial I}{\partial x} \frac{dx}{dt} + \frac{\partial I}{\partial y} \frac{dy}{dt} = - \frac{\partial I}{\partial t}

    * We can resolve v thanks to least squares method  :

    .. math::

        v = (A^T A)^{-1}A^T B

    * Lucas & Kanade works on a pixel and his neighbourhood so :

    .. math::

        A =
            \left(\begin{array}{cc}
            I_x(q1) & I_y(q1)\\
            I_x(q2) & I_y(q2) \\
            . & . \\
            . & . \\
            . & . \\
            I_x(qn) & I_y(qn)
            \end{array}\right)

        v =
            \left(\begin{array}{cc}
            V_x\\
            V_y
            \end{array}\right)


        B =
            \left(\begin{array}{cc}
            -I_t(q1) \\
            -I_t(q2)  \\
            .  \\
            .  \\
            .  \\
            -I_t(qn)
            \end{array}\right)



Configuration and parameters
----------------------------
+---------------------+-------------------+--------+---------------+--------------------------------+----------+
| Name                | Description       | Type   | Default value | Available value                | Required |
+=====================+===================+========+===============+================================+==========+
| *refinement_method* | Refinement method | string |               |"interpolation", "optical_flow" | No       |
+---------------------+-------------------+--------+---------------+--------------------------------+----------+

**Example**

.. sourcecode:: text

    {
        "input" :
        {
            ...
        },
        "pipeline" :
        {
            ...
            "refinement":
            {
               "refinement_method": "interpolation"
            }
            ...
        }
    }

