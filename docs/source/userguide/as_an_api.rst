.. _as_an_api:

As an API
=========

Pandora2D API usage
*******************

Pandora2D provides a full python API which can be used to compute disparity maps as shown in this basic example:

.. sourcecode:: python

    import numpy as np
    import pandora
    import pandora2d

    from pandora2d.state_machine import Pandora2DMachine
    from pandora2d import check_configuration, common
    from pandora2d.img_tools import create_datasets_from_inputs

    # Paths to left and right images
    img_left_path = "data/left.tif"
    img_right_path = "data/right.tif"

    user_cfg = {
     # image configuration
        'input': {
            'left': {
                'img': img_left_path,
                'nodata': np.nan,
            },
            'right': {
                'img': img_right_path,
                'nodata': np.nan,
            },
            "col_disparity": {"init": 0, "range": 2},
            "row_disparity": {"init": 0, "range": 2},
        },
        # define pipeline configuration
         'pipeline':{
            "matching_cost" : {
                "matching_cost_method": "zncc",
                "window_size": 5,
            },
            "disparity": {
                "disparity_method": "wta",
                "invalid_disparity": -9999
            },
            "refinement" : {
                "refinement_method" : "optical_flow"
            }
        },
        "output": {
            "path": "as_an_api_output"
        },
    }

    # read images
    image_datasets = create_datasets_from_inputs(input_config=user_cfg["input"], estimation_cfg=user_cfg["pipeline"].get("estimation"))

    # instantiate Pandora2D Machine
    pandora2d_machine = Pandora2DMachine()

    # check the configurations and sequences steps
    pipeline_cfg = check_configuration.check_pipeline_section(user_cfg, pandora2d_machine)

    # prepare the machine
    pandora2d_machine.run_prepare(
        image_datasets.left,
        image_datasets.right,
        user_cfg
        )

    # trigger all the steps of the machine at ones
    dataset, completed_cfg = pandora2d.run(
        pandora2d_machine,
        image_datasets.left,
        image_datasets.right,
        pipeline_cfg
        )

    # save dataset
    common.save_disparity_maps(dataset, completed_cfg)


Pandora2D's data
****************

Images
######

Pandora2D reads the input images before the stereo computation and creates two datasets, one for the left and one for the right
image which contain the data's image and additional information.

Example of an image dataset

::

    Dimensions:  (col: 450, row: 375)
    Coordinates:
      * col      (col) int64 0 1 2 3 4 5 6 7 8 ... 442 443 444 445 446 447 448 449
      * row      (row) int64 0 1 2 3 4 5 6 7 8 ... 367 368 369 370 371 372 373 374
      * band_disp               (band_disp) <U3 'min' 'max'
    Data variables:
        im       (row, col) float32 88.0 85.0 84.0 83.0 ... 176.0 180.0 165.0 172.0
        msk      (row, col) int16 0 0 0 0 0 0 0 0 0 0 0 0 ... 0 0 0 0 0 0 0 0 0 0 0
        col_disparity (band_disp, row, col) int64 -2 -2 -2 -2 ... 2 2 2 2
        row_disparity (band_disp, row, col) int64 -2 -2 -2 -2 ... 2 2 2 2
    Attributes:
        no_data_img:   0
        crs:           None
        transform:     | 1.00, 0.00, 0.00|| 0.00, 1.00, 0.00|| 0.00, 0.00, 1.00|
        valid_pixels:  0
        no_data_mask:  1
        col_disparity_source:  [-2, 2]
        row_disparity_source:  [-2, 2]

    Two data variables are created in this dataset:

    * *im*: contains input image data
    * *msk*: contains input mask data + no_data of input image

.. note::
    This example comes from a dataset created by Pandora's reading function. Dataset attributes
    *valid_pixels* and *no_data_mask* cannot be modified with this function, as they are defined by the *msk*
    data convention.
    For an API user who wants to create its own dataset without using Pandora's reading function, it is
    possible to declare its own mask convention with these attributes:

      * *no_data_img* : value of no_data in input image
      * *valid_pixels*: value of valid pixels in input mask
      * *no_data_mask*: value of no_data pixel in input mask


Cost volumes
############
Pandora2D will then store all the cost volumes together in a 4D (dims: row, col, disp_col, disp_row)
xarray.DataArray named cost_volumes. When matching is impossible, the matching cost is set to np.nan.

::

    <xarray.Dataset> Size: 224B
    Dimensions:       (row: 3, col: 3, disp_row: 2, disp_col: 2)
    Coordinates:
      * col           (col) int64 24B 0 1 2
      * row           (row) int64 24B 0 1 2
      * disp_row      (disp_row) int64 16B -1 0
      * disp_col      (disp_col) int64 16B -1 0
    Data variables:
        cost_volumes  (row, col, disp_row, disp_col) float32 144B nan nan ... 4.0
    Attributes: (12/16)
        no_data_img:           -9999
        valid_pixels:          0
        no_data_mask:          1
        crs:                   None
        transform:             | 1.00, 0.00, 0.00|\n| 0.00, 1.00, 0.00|\n| 0.00, ...
        col_disparity_source:  [-1, 3]
        ...                    ...
        offset_row_col:        0
        measure:               sad
        type_measure:          min
        cmax:                  10004
        disparity_margins:     None
        step:                  [1, 1]

Disparity map
#############

The *Disparity computation* step generates two disparity maps in cost volume geometry. One named **row_map** for the
vertical disparity and one named **col_map** for the horizontal disparity. These maps are float32 type 2D xarray.DataArray,
stored in a xarray.Dataset. 

This xr.Dataset also contains the **validity maps** stored in uint8: 

    * A global validity map 'validity_mask' indicating whether each point is valid (value 0), partially valid (value 1) or invalid (value 2).
    * A map for each criteria, indicating for each point whether the corresponding criteria has been raised at that point (value 0) or not (value 1).

::

    <xarray.Dataset>
    Dimensions:  (col: 450, row: 375, criteria: 2)
    Coordinates:
      * row      (row) int64 0 1 2 3 4 5 6 7 8 ... 367 368 369 370 371 372 373 374
      * col      (col) int64 0 1 2 3 4 5 6 7 8 ... 442 443 444 445 446 447 448 449
      * criteria (criteria) <U43 'validity_mask' ... 'PANDORA2D_M...
    Data variables:
        row_map  (row, col) float32 nan nan nan nan nan nan ... nan nan nan nan nan
        col_map  (row, col) float32 nan nan nan nan nan nan ... nan nan nan nan nan
        correlation_score  (row, col) float32 nan nan nan nan nan nan ... nan nan nan nan nan
        validity  (row, col, criteria) uint8 0 1 0 0 2 0 ... 0 1 0 0 0
    Attributes:
        offset:       {'row': 0, 'col': 0}
        step:         {'row': 1, 'col': 1}
        invalid_disp: -9999
        crs:          None
        transform:    | 1.00, 0.00, 0.00|| 0.00, 1.00, 0.00|| 0.00, 0.00, 1.00|

.. warning::
    The validity maps are not yet operational as development is still in progress.

Border management
#################


Left image
----------

Pixels of the left image for which the measurement thumbnail protrudes from the left image are set to :math:`nan`
on the cost volume.
For a similarity measurement with a 5x5 window, these incalculable pixels in the left image correspond
to a 2-pixel crown at the top, bottom, right and left, and are represented by the offset_row_col attribute in
the xarray.Dataset.

Right image
-----------

Because of the disparity range choice, it is possible that there is no available point to scan on the right image.
In this case, matching cost cannot be computed for this pixel and the value will be set to :math:`nan` .
Then bit 1 will be set : *The point is invalid: the disparity interval to explore is
absent in the right image* and the point disparity will be set to *invalid_disparity*.
Moreover, everytime Pandora2D shifts the right image it introduces a new line set at *nodata_right* value. The matching
cost cannot be computed for this line to.
