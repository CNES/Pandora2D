.. _faq:

FAQ
=========

How do I use a step parameter in the matching cost computation?
****************************************************************

**A notebook showing the use of the step parameter and Region Of Interest (ROI) is available:**

It is possible to add a step parameter in the configuration file. This parameter ensures that not all pixels are calculated during the matching cost computation.

.. code:: ipython3

    user_cfg = {
        "input": {
            "left": {
                "img": img_left_path,
                "nodata": "NaN",
            },
            "right": {
                "img": img_right_path,
                "nodata": "NaN",
            },
            "col_disparity": [-3, 3],
            "row_disparity": [-3, 3],
        },
        "pipeline": {
            "matching_cost": {
                "matching_cost_method": "zncc",
                "window_size": 7,
                "step": [5, 5],
            },
            "disparity": {
                "disparity_method": "wta",
                "invalid_disparity": -9999,
            },
            "refinement": {
                "refinement_method": "interpolation",
            },
        },
    }

How do I choose to process only a certain part of the image? 
****************************************************************

**A notebook showing the use of the step parameter and Region Of Interest (ROI) is available:**

It is possible to work on only one section of the image with an ROI. For this, the user can specify the area he wants in the configuration file. 

.. code:: ipython3

    user_cfg = {
        "input": {
            "left": {
                "img": img_left_path,
                "nodata": "NaN",
            },
            "right": {
                "img": img_right_path,
                "nodata": "NaN",
            },
            "col_disparity": [-3, 3],
            "row_disparity": [-3, 3],
        },
        "ROI": {
            "col": {"first": 10, "last": 100},
            "row": {"first": 10, "last": 100},
        },
        "pipeline": {
            "matching_cost": {
                "matching_cost_method": "zncc",
                "window_size": 7,
                "step": [5, 5],
            },
            "disparity": {
                "disparity_method": "wta",
                "invalid_disparity": -9999,
            },
            "refinement": {
                "refinement_method": "interpolation",
            },
        },
    }

.. code:: ipython3

    user_cfg["ROI"]["margins"] = pandora2d_machine.global_margins.astuple()
    roi = get_roi_processing(user_cfg["ROI"], user_cfg["input"]["col_disparity"], user_cfg["input"]["row_disparity"])

.. code:: ipython3

    img_left = create_dataset_from_inputs(input_config=input_config["left"], roi=roi)
    img_right = create_dataset_from_inputs(input_config=input_config["right"], roi=roi)

.. note::
    When the usage_step_roi_config.ipynb notebook is run, disparity maps are displayed. 
    Margins can be present on these disparity maps, which is why they may be larger than the ROI given by the user. 
    To remove these margins and display only the user ROI, you can use the `pandora2d.img_tools.remove_roi_margins()` method. 
