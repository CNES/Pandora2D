Getting started
===============

Overview
########

Pandora2d  is a coregistration tool that provide disparity maps for images pairs with a combination in both direction.
It uses `Pandora <https://github.com/CNES/Pandora>`__ who works with stereo pair of images only.


Install
#######
Pandora2D is available on Gitlab and can be installed by:

.. code-block:: bash

    #install pandora2d latest release
    pip install pandora2d

First step
##########

Pandora2d requires a `config.json` input file to declare the pipeline and the pair of images to process.
Download our data sample to start right away !

.. code-block:: bash

    # Images pairs with a combination of vertical and horizontal stereo
    wget https://raw.githubusercontent.com/CNES/Pandora/master/data_samples/images/maricopa.zip
    # Config file
    wget https://raw.githubusercontent.com/CNES/Pandora/master/data_samples/json_conf_files/a_basic_pipeline.json
    #uncompress data
    unzip maricopa.zip
    #run Pandora2D
    pandora2d a_basic_pipeline.json output_dir

Customize
#########

To create your own coregistration pipeline and choose among the variety of
algorithms we provide, please consult :ref:`userguide`.

You will learn:

    * which steps you can use and combine
    * how to quickly set up a Pandora2D pipeline

Credits
#######

Pandora2D uses `transitions <https://github.com/pytransitions/transitions>`_ to manage the pipelines one can create.
Images I/O are provided by `rasterio <https://github.com/mapbox/rasterio>`_ and we use `xarray <https://github.com/pydata/xarray>`_
to handle 3D Cost Volumes with few `multiprocessing <https://github.com/uqfoundation/multiprocess>`_ optimisations.

Our data test sample is based on the Peps Sentinel2 website (by CNES).

Related
#######

* Pandora2D calls Pandora N times for N row disparities.
    `Pandora <https://github.com/cnes/pandora>`_ - stereo matching framework