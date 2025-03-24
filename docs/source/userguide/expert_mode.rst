.. _Expert_mode:

Expert mode
===========

Resume
******

The profiling expert mode is intended for users who want to measure the performance of Pandora2D on their personal computer.
In the output folder, they can obtain a number of charts that calculate averages and other metrics for each step throughout the executions.

How to profile more functions ?
*******************************


This option requires the user to be familiar with the pandora2d code.

First, when they activate the `expert_mode` key in the configuration, they have access by default to performance
information related to each stage of the state machine.
All data is stored in the code in a `pandas.DataFrame` and locally in a CSV file, then presented as a graph in a PDF file.

If the user wants to analyze the performance of another function, they can add the decorator
`@mem_time_profile_profile(name="Function name")` above that function.
If they want to obtain more metrics, they need to add them to the "metrics_list" in the `profiling.py` file.

The graphs are handled by the `generate_figure` function.

.. note::
    Profiling certain functions can significantly increase execution times.



Parameters and configuration :
##############################

Expert mode profiling section is composed of the following keys:

.. list-table:: Expert mode section
    :header-rows: 1

    * - Name
      - Description
      - Type
      - Default value
      - Required
    * - *folder_name*
      - path where to save profiling informations
      - string
      - None
      - Yes

**Example**

.. code:: json
    :name: Input example

    {
        "input":
        {
            // inputs content
        }
        ,
        "pipeline" :
        {
            // pipeline content
        },
        "expert_mode":
        {
            "profiling":
            {
                "folder_name": "profiling_output"
        }
        "output": {
            "path": "expert_mode_output"
        },
    }
