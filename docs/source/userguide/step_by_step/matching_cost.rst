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
       | "float64", "f8", "d" for "zncc" or "mutual_information"
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
    Two implementations of zncc are available for matching cost method: one in C++ and one in Python. 
    By default, the C++ zncc is used when using the “zncc” matching cost method. 
    To use the Python version, enter “zncc_python” as the matching cost method in the configuration file.

.. warning::
    The subpix parameter can only take values 1, 2 and 4.

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