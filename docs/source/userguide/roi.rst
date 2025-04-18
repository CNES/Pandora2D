.. _roi:

Region of interest
==================

It is possible to work on only one section of the image with an ROI. For this, the user can specify the area he wants
in the configuration file. At the moment, our only option is to work with a sensor ROI.


Configuration and parameters
****************************

.. list-table:: ROI parameters
   :widths: 25 25 25 25 25
   :header-rows: 1


   * - Name
     - Description
     - Type
     - Default value
     - Required
   * - **col**
     - Columns sensor coordinates
     - dict
     - None
     - No
   * - **row**
     - Rows sensor coordinates
     - dict
     - None
     - No

Row and col parameters contain dictionaries, each is different depending on the mode used.

- Sensor mode : The user uses pixel coordinates to select an ROI.


.. list-table::
   :widths: 19 19 19 19 19 19
   :header-rows: 1


   * - Mode
     - Name
     - Description
     - Type
     - Default value
     - Required
   * - **sensor**
     - first
     - First pixel in selected coordinates
     - int
     -
     - Yes, if ROI is desired.
   * - **sensor**
     - last
     - Last pixel in selected coordinates
     - int
     -
     - Yes, if ROI is desired.

.. note::
    It is necessary for the **first** coordinate to be lower than the **last** coordinate.

**Example**

.. code:: json
    :name: ROI example

    {
        "input":
        {
            // input content
        },
        "ROI":
        {
            "col": {"first": 10, "last": 100},
            "row": {"first": 10, "last": 100}
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

