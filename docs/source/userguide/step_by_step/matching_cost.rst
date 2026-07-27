.. _matching_cost:

Matching cost computation
=========================

Theoretical basics
------------------

Theory is detailed in the exploring the field section (:ref:`exploring_the_field_matching_cost`).

Configuration and parameters
----------------------------

.. list-table:: Available parameters
   :name: matching_cost available parameters
   :widths: 19 19 19 19 19 19
   :header-rows: 1


   * - Name
     - Description
     - Type
     - Default value
     - Available value
     - Required
   * - matching_cost_method
     - Similarity measure
     - string
     - None
     - * "ssd"
       * "sad"
       * "zncc"
       * "zncc-optim-1"
       * "zncc-optim-2"
       * "zncc_python"
       * "mc_cnn"
       * "mutual_information"
     - Yes
   * - window_size
     - Window size for similarity measure
     - int
     - 5
     - | > 0 and **odd**
       | or 11 if "matching_cost_method" is "mc_cnn"
       | or >1 if "refinement_method" is "optical_flow"
     - No
   * - step
     - Step [row, col] for computing similarity coefficient
     - list[int, int]
     - [1, 1]
     - list[int >0, int >0]
     - No
   * - subpix
     - Subpix parameter for computing subpixel disparities
     - int
     - 1
     - [1,2,4]
     - No
   * - float_precision
     - Precision used to compute cost volumes
     - string
     - "float32"
     - | "float32", "f4", "f" for all methods
       | "float64", "f8", "d" for "zncc" (including "zncc-optim-1" and "zncc-optim-2") or "mutual_information"
     - No
   * - spline_order
     - Spline order used for interpolation when subpix > 1
     - int
     - 1
     - > 0 and < 6
     - No


.. note::
    The order of steps should be [row, col].

.. note::
    Many implementations of zncc are available for matching cost method: two in C++ and one in Python.
    When using the ``zncc`` matching cost method, Pandora2D automatically selects the most appropriate C++ implementation
    from ``window_size`` and ``step``.
    Pandora2D chooses ``zncc-optim-1`` when ``window_size / max(step_row, step_col) > 3``, otherwise ``zncc-optim-2``.
    This threshold was determined empirically by measuring the execution time of both implementations on various
    ``window_size``/``step`` configurations.
    To force a specific C++ implementation, use ``zncc-optim-1`` or ``zncc-optim-2`` in the configuration file.
    To use the Python version, enter ``zncc_python`` as the matching cost method in the configuration file.

.. note::
    In C++, two versions of the ``zncc`` method are proposed.
    We recommend to use the first optimization ``zncc-optim-1`` when all pixels are computed (step is small) and when the disparity range is small.
    We recommend the second optimization ``zncc-optim-2`` when an important step is provided and disparity ranges are large.

.. note::
    To use ``mc_cnn`` as ``matching_cost_method``, the MCCNN plugin must be installed first.
    Install it with:

    .. code-block:: bash

        pip install pandora-plugin-mccnn

    See `Pandora plugin documentation <https://pandora.readthedocs.io/en/stable/userguide/plugins/plugin_mccnn.html>`_.

.. warning::
    The subpix parameter can only take values 1, 2 and 4.

.. warning::
    When ``matching_cost_method`` is set to ``zncc``, the C++ implementation is selected automatically but can still be
    forced with ``zncc-optim-1`` or ``zncc-optim-2``.

**Example**

.. sourcecode:: json
    :name: matching_cost example

    {
        "input" :
        {
            // input content
        },
        "pipeline" :
        {
            //...
            "matching_cost":
            {
                "matching_cost_method": "ssd",
                "window_size": 7,
                "step" : [5, 5],
                "subpix": 4,
            },
            //...
        },
        "output":
        {
            //...
        }
    }
