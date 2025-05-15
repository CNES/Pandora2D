.. _segment_mode:

Segment mode
============

Large images can be processed using the segment mode. 
This mode independently divides the image into sections and processes them sequentially. 
It then reconstructs the result as if the image had been processed conventionally in Pandora2d.


.. warning::
    This mode is still under development. It will be available in the next release : 0.6.0.

Configuration and parameters
****************************

.. list-table:: Segment_mode parameters
   :widths: 25 25 25 25 25
   :header-rows: 1


   * - Name
     - Description
     - Type
     - Available value
     - Required
   * - **enable**
     - Activate mode
     - bool
     - True/False
     - No
   * - **memory_per_work**
     - Reserved memory for pandora2d calculations (Mb)
     - int
     - > 0 
     - No

**Example**

.. code:: json
    :name: Segment_mode example

    {
        "input":
        {
            // input content
        },
        "segment_mode": {
            "enable": true,
            "memory_per_work": 4000
        },
        "ROI":
        {
            // ROI content
        },
        "pipeline" :
        {
            // pipeline content
        },
        "output" :
        {
            // output content
        }
    }
