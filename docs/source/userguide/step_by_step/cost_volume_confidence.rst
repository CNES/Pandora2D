.. _cost_volume_confidence:

Cost volume confidence
======================

The purpose of this step is to compute confidence measure on the cost volume.

.. note::
    This feature is based on the ambiguity calculated by `Pandora <https://pandora.readthedocs.io/en/stable/userguide/step_by_step/cost_volume_confidence.html#theoretical-basics>`_

.. warning::
    This initial version, available in Pandora2d 1.1.0, should not be used with Pandora correlation metrics (sad, ssd, zncc_python, mc_cnn). 
    An update in a future version will resolve this issue.

Configuration and parameters
----------------------------

.. list-table:: Configuration and parameters
   :widths: 19 19 19 19 19 19
   :header-rows: 1


   * - Name
     - Description
     - Type
     - Default value
     - Available value
     - Required
   * - *confidence_method*
     - Cost volume confidence method
     - str
     -
     - | "ambiguity"
     - Yes
   * - *eta_max*
     - Maximum :math:`\eta`
     - float
     - 0.7
     - >0 and <1
     - No
   * - *eta_step*
     - :math:`\eta` step
     - float
     - 0.01
     - >0 and <1
     - No
   * - *normalization*
     - Ambiguity normalization
     - bool
     - true
     - true, false
     - No

**Example**

.. sourcecode:: json

    {
        "input" :
        {
            // ...
        },
        "pipeline" :
        {
            // ...
            "cost_volume_confidence":
            {
                "confidence_method": "ambiguity",
                "eta_max": 0.7,
                "eta_step": 0.01
            },
            // ...
        }
    }
