.. _validity_criteria:

.. |left image criterion note| replace:: This criterion results from left image and is not dependent upon disparity.
                                         As a result, this criterion will be assigned to all disparities associated to this pixel in the criteria data array.

Validity criteria
=================

Pandora 2D utilizes seven criteria to either invalidate pixels that cannot be computed or identify specific pixels.
During execution, these criteria are represented as 8-bit encoded flags and stored as a criteria data variable named `criteria` within the cost volumes.
This data variable shares the same dimensions as the cost volumes: `row`, `col`, `disp_row`, and `disp_col`.

This approach enables the assignment of a criterion to each disparity pair for every pixel in the image, ensuring a clear representation of the underlying process.

.. note:: The criteria are cumulative, meaning multiple criteria can be applied to a single pixel—except at the image boundaries.

For ease of use, criterion values are managed as an enumeration (enum), allowing operations to be performed via their names rather than directly manipulating numerical values.

Ultimately, the criteria dataset serves as the foundation for constructing the validity dataset,
which provides information regarding the reliability of each pixel in the image.

Below is a summary of each criterion (clic on criterion's name to jump to its detailed description).

.. list-table:: Criteria summary
   :header-rows: 1


   * - **Criterion**
     - **Value in criteria data var**
     - **Description**
   * - :ref:`P2D_LEFT_BORDER_explanation`
     - 2⁰ = 1
     - border of left image according to window size
   * - :ref:`P2D_LEFT_NODATA_explanation`
     - 2¹ = 2
     - nodata in left mask
   * - :ref:`P2D_RIGHT_NODATA_explanation`
     - 2² = 4
     - nodata in right mask
   * - :ref:`P2D_RIGHT_DISPARITY_OUTSIDE_explanation`
     - 2³ = 8
     - disparity is out the right image
   * - :ref:`P2D_INVALID_MASK_LEFT_explanation`
     - 2⁴ = 16
     - invalidated by validity mask of left image
   * - :ref:`P2D_INVALID_MASK_RIGHT_explanation`
     - 2⁵ = 32
     - invalidated by validity mask of right image
   * - :ref:`P2D_PEAK_ON_EDGE_explanation`
     - 2⁶ = 64
     - | The correlation peak is at the edge of disparity range.
       | The calculations stopped at the pixellic stage.



Detailed Criteria's explanations
--------------------------------

.. _P2D_LEFT_BORDER_explanation:

P2D_LEFT_BORDER
^^^^^^^^^^^^^^^

This criterion excludes edge pixels where the matching cost window would extend
beyond the image boundaries. Since this window is centered on the pixel and has
an odd size, the offset considered follows this formula:

:math:`offset = \lfloor (window\_ size - 1) / 2 \rfloor`

Where window_size is an :ref:`input parameter of matching cost<matching_cost available parameters>`.

.. note:: |left image criterion note|

.. warning:: When this criterion is raised for a pixel, no other criterion is raised for this pixel.


.. image:: ./Images/criteria/left_border_criteria.drawio.svg
    :align: center

|
|

.. _P2D_LEFT_NODATA_explanation:

P2D_LEFT_NODATA
^^^^^^^^^^^^^^^

This criterion is raised when ``NO DATA`` is found in the matching cost window.
To identify the affected pixels, a binary dilation, equivalent in size to the
matching cost window, is applied to the pixels corresponding to ``NO DATA`` in
the left image mask.

.. note:: |left image criterion note|

.. image:: ./Images/criteria/left_nodata_criteria.drawio.svg
    :align: center

|
|

.. _P2D_RIGHT_NODATA_explanation:

P2D_RIGHT_NODATA
^^^^^^^^^^^^^^^^

This criterion is raised when a pixel within the matching cost window of the
explored pixel in the right image is marked as ``NO DATA`` in the right mask.

.. container:: html-image

    .. raw:: html
        :file: ./Images/criteria/right_nodata_criteria.drawio.html

|
|

.. _P2D_RIGHT_DISPARITY_OUTSIDE_explanation:

P2D_RIGHT_DISPARITY_OUTSIDE
^^^^^^^^^^^^^^^^^^^^^^^^^^^

This criterion is raised when the matching cost window goes outside the right image.

.. container:: html-image

    .. raw:: html
        :file: ./Images/criteria/right_disparity_outside_criteria.drawio.html

|
|

.. _P2D_INVALID_MASK_LEFT_explanation:

P2D_INVALID_MASK_LEFT
^^^^^^^^^^^^^^^^^^^^^

This criterion is raised when the pixel is marked as INVALID in the left mask.

.. note:: |left image criterion note|

.. image:: ./Images/criteria/left_invalid_mask_criteria.drawio.svg
    :align: center

|
|

.. _P2D_INVALID_MASK_RIGHT_explanation:

P2D_INVALID_MASK_RIGHT
^^^^^^^^^^^^^^^^^^^^^^

This criterion is raised when the explored pixel in the right image is marked as INVALID in the right mask.

.. container:: html-image

    .. raw:: html
        :file: ./Images/criteria/invalid_mask_right_criteria.drawio.html

|
|

.. _P2D_PEAK_ON_EDGE_explanation:

P2D_PEAK_ON_EDGE
^^^^^^^^^^^^^^^^

The best similarity measure selected by the Winner Takes All algorithm is an extremum.
If the selected extremum is on the edge of the cost surface, we cannot determine whether it is a true extremum
or merely point of a trend cut off by the chosen disparity range.

.. image:: ./Images/criteria/peak_on_edge_trend.drawio.svg
    :align: center

In this case, the P2D_PEAK_ON_EDGE criterion is raised.

.. image:: ./Images/criteria/peak_on_edge_criteria.drawio.svg
    :align: center

|
|

Validity dataset
----------------

From the ``criteria`` data array, the ``validity`` dataset is generated to
summarize the information and store it. It consists of multiple data variables,
one for each criterion, along with the validity data variable. These variables
are saved as separate bands, named after their corresponding criterion, in a TIFF
file named ``validity.tif``, located in the ``output/disparity_map`` directory.

Criterion bands
^^^^^^^^^^^^^^^

A separate band is created for each criterion. If a criterion is present in the
cost surface mask of a given pixel in the image, the corresponding pixel in the
band is set to ``1``; otherwise, it is set to ``0``.

Validity
^^^^^^^^

If at least one criterion is present in the cost surface mask of a pixel, this pixel is considered as ``PARTIALLY_INVALID``
except if all the disparity pairs contain a criterion: in this case, the pixel is considered as ``INVALID``.

In the validity band, ``VALID`` pixels (those without any raised criterion) are
encoded as ``0``, ``PARTIALLY_INVALID`` pixels are encoded as ``1``, and
``INVALID`` pixels are encoded as ``2``.
