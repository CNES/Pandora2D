.. _estimation:

Estimation computation
=========================

This step aims to calculate a steady shift in both row and column to establish the starting central
point of the disparity intervals that are applied in the matching_cost process.

**Phase cross correlation method**

The phase cross correlation method depends on frequency domain.
It isolates the phase information of cross-correlation from two similar images.

The phase cross correlation algorithm is divided into 4 steps:

- Firstly, we compute the Fourier transform of both the right and left images.
- Secondly, we calculate the cross-correlation between the two Fourier transforms.
- Then, we identify the maximum peak and retrieve a pixel-level shift.
- Finally, for sub-pixel level shifting, we perform an interpolation around this peak.

.. note:: Currently, only the **phase_cross_correlation** method is implemented in Pandora2d.
          We use the phase_cross_correlation function from scipy, for further information please see
          `Scipy documentation <https://scikit-image.org/docs/stable/api/skimage.registration.html#skimage.registration.phase_cross_correlation>`__


Configuration and parameters
----------------------------
.. warning::

    Disparities must not be specified in the configuration file if you are using estimation.

.. list-table:: Parameters
    :header-rows: 1


    * - Name
      - Description
      - Type
      - Default value
      - Available value
      - Required
    * - *estimation_method*
      - estimation measure
      - string
      - None
      - "phase_cross_correlation"
      - Yes
    * - *range_col*
      - Exploration around the initial disparity for columns
      - int
      - 5
      - >2, odd number
      - No
    * - *range_row*
      - Exploration around the initial disparity for rows
      - int
      - 5
      - >2, odd number
      - No
    * - *sample_factor*
      - | Upsampling factor.
        | Images will be registered to within 1 / upsample_factor of a pixel
      - int
      - 1
      - >= 1, multiple of 10
      - No


**Example**

.. code:: json
    :name: Estimation example

    {
        "input":
        {
            // ...
        },
        "pipeline":
        {
            "estimation":
            {
                "estimation_method": "phase_cross_correlation",
                "range_col": 5,
                "range_row": 5,
                "sample_factor": 20
            }
            // ...
        },
        "output":
        {
            // ...
        }
    }


Outputs:
--------

- Showed in log in verbose mode
- Written in the output configuration file
- Stored in the inputs_dataset