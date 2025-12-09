.. _exploring_the_field_multiscale_mode:

Multiscale mode 
===============

The pandora2d multiscale mode allows fine disparity maps to be calculated. 

To do this, disparities are computed on an image pyramid with different resolutions. 
Then, for each resolution, row and column initial disparity grids are calculated by estimating local or global deformation models 
based on the disparity maps calculated at the previous resolution. 

This mode has two advantages: 

- it speeds up the pandora2d pipeline by estimating a priori values for the initial disparity for each point.
- it makes pandora2d more robust for certain types of images by using image subsampling.

Estimation of deformation models
--------------------------------

Currently, the row and column deformation models are estimated using a polynomial whose degree can be chosen. 
This polynomial is estimated by solving a least squares problem on the positions of the points in the image.

The solved least squares problem is defined as follows: 

.. math::
   
   \min_{\beta} \; \| y - X \beta \|_2^2

where :

- :math:`y \in \mathbb{R}^n` is the vector of :math:`n` valid final positions in row **or** in columns (initial positions to which the disparities calculated by pandora2d have been added).
- :math:`\beta \in \mathbb{R}^p` is the vector of coefficients that we want to estimate in order to determine the deformation model. Its size :math:`p` depends on the degree of the polynomial.
- :math:`X \in \mathbb{R}^{n \times p}` is the design matrix up to a given degree. It generates all polynomial combination of the initial positions in rows and columns.

Cholesky method is used to resolve these least squares problems. 

A deformation model can be calculated for the entire image, 
or several deformation models can be calculated by dividing the image into meshes. 
The mesh division is parameterized in the multiscale :ref:`configuration <multiscale_mode>`.

The coefficients :math:`\beta` are then used to calculate the grid deformation which is then converted into initial disparity grids in rows and columns for the next resolution.

Example of multiscale pipeline
------------------------------

Below are step-by-step diagrams explaining how multiscale mode works. 

.. note:: 
    In this example, we consider the estimation of a row deformation model. 
    The same steps apply to estimating a column deformation model.

.. important::
    If you choose to process the image by mesh, each step below is performed for each mesh.  

.. container:: html-image

    .. raw:: html
        :file: ./Images/multiscale_mode.drawio.html
