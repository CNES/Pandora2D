# Definition

A non-regression test below is used to validate each function of the Pandora2D tool and to ensure that future developments will not lead back to an earlier stage.

# Functionality
For each functionality, a directory is created. Theses functionalities are as follows.

## Target Grid
The target grid is the processing area in pandora2d. This can be an area or certain pixels. The sub-functionalities tested here are:

+-------------------+---------------------------------------------------------------------------------+
| Sub-functionality | Description                                                                     |
+-------------------+---------------------------------------------------------------------------------+
| roi               | A ROI (region of interest) is a polygon representing the image to be processed. |
+-------------------+---------------------------------------------------------------------------------+
| step              | A step can be used to avoid processing all the pixels.                          |
+-------------------+---------------------------------------------------------------------------------+

## Mode
The mode is the type of search (example : disparity range). The sub-functionalities tested here are:

+------------------------+---------------------------------------------------------------------+
| Sub-functionality      | Description                                                         |
+------------------------+---------------------------------------------------------------------+
| multiscale             | An image pyramid is computed according to the given scale factor    |
+------------------------+---------------------------------------------------------------------+
| strip                  | A large image cut into sub-images with the roi                      |
+------------------------+---------------------------------------------------------------------+
| initial_disparity      | A search with a fixed pixel disparity                               |
+------------------------+---------------------------------------------------------------------+
| initial_disparity_grid | A search with disparity grid, i.e. a fixed disparity for each pixel |
+------------------------+---------------------------------------------------------------------+
| exploration_pix        | A research zone around a disparity                                  |
+------------------------+---------------------------------------------------------------------+
| exploration_grid       | A research zone grid, i.e. an exploration for each pixel            |
+------------------------+---------------------------------------------------------------------+
| subpix                 | A more accurate search for disparities                              |
+------------------------+---------------------------------------------------------------------+

## Criteria
Invalidity indicators are raised depending on the calculation on the pixel in question (use of masks, area of disparity too large, etc.). The sub-functionalities tested here are:

+----------------------+-----------------------------------------------------------------+
| Sub-functionality    | Description                                                     |
+----------------------+-----------------------------------------------------------------+
| left_mask            | Using a mask on the left image                                  |
+----------------------+-----------------------------------------------------------------+
| right_mask           | Using a mask on the right image                                 |
+----------------------+-----------------------------------------------------------------+
| no_data_left         | Some pixels contain no data on the left image                   |
+----------------------+-----------------------------------------------------------------+
| no_data_right        | Some pixels contain no data on the right image                  |
+----------------------+-----------------------------------------------------------------+
| disparity_range      | The disparity search zone is too large                          |
+----------------------+-----------------------------------------------------------------+
| matching_cost_window | The window size of matching cost algorithm is outside the image |
+----------------------+-----------------------------------------------------------------+

## Matching_cost
This is the stage where a similarity score is calculated between the two images. The algorithms tested here are:

+-----------+----------------------------------------+
| Algorithm | Description                            |
+-----------+----------------------------------------+
| zncc      | Zero mean Normalized Cross Correlation |
+-----------+----------------------------------------+
| sad       | Sum of Absolute Differences            |
+-----------+----------------------------------------+
| ssd       | Sum of Squared Differences             |
+-----------+----------------------------------------+
| MI        | Mutual Information                     |
+-----------+----------------------------------------+

## Disparity
Selecting the best similarity score, for the moment there is only the WTA method (Winner takes all).

## Refinement
Accurate the disparity to smooth outliers. The algorithms tested here are:

+-----------+-----------------------------------------+
| Algorithm | Description                             |
+-----------+-----------------------------------------+
| fo        | Optical flow                            |
+-----------+-----------------------------------------+
| dichotomy_python | Reseach by dichotomy with python |
+-----------+-----------------------------------------+
| bicubic   | Bicubic interpolation                   |
+-----------+-----------------------------------------+
| sinc      | Sinux cardinal                          |
+-----------+-----------------------------------------+
| spline    | Spline                                  |
+-----------+-----------------------------------------+

## Validation
A criterion that gives the user additional confidence in the result. The algorithms tested here are:

+---------------------+----------------------------------+
| Algorithm           | Description                      |
+---------------------+----------------------------------+
| validation_fast     | as its name suggests             |
+---------------------+----------------------------------+
| validation_accurate | as its name suggests             |
+---------------------+----------------------------------+
| border_pic          | Ambiguity indicator              |
+---------------------+----------------------------------+
| fo_diverge          | in connection with optical flows |
+---------------------+----------------------------------+
| fo_hess             | in connection with optical flows |
+---------------------+----------------------------------+
| fo_eigenval         | in connection with optical flows |
+---------------------+----------------------------------+