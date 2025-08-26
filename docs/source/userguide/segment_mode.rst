.. _segment_mode:

Segment mode
============

Large images can be processed using the segment mode. 
This mode independently divides the image into segments (depending on the memory allocated by the user to pandora2d) and processes them sequentially. 
It then reconstructs the result as if the image had been processed conventionally in Pandora2d.


.. note::
    If the memory allocated to pandora2d by the user is not sufficient to process the smallest possible segment (a single line of the image), 
    an error message indicating the minimum memory required is returned.

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
     - Reserved memory for pandora2d calculations (MB)
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
