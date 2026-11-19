.. _skip_mode:

Skip mode
=========

Skip mode applies rules that skip some calculations and/or keep extra information
about the pipeline computations.
The ``skip_mode`` section declares the skip modes and their associated rules.
Their effect on the pipeline will be added with each rule.

The section is optional and sits at the same level as ``segment_mode`` or the ROI.
When it is absent, no skip mode is active.

.. warning::
   **Skip mode is still in development.** Additional rules may have restrictions
   or be adjusted as implementation progresses.

.. warning::
   The ``incomplete_disparity_range`` rule is not compatible with Pandora matching cost
   methods (``sad``, ``ssd``, ``zncc_python``, ``mc_cnn``). These metrics are computed outside
   the C++ matching cost implementation.


Configuration and parameters
****************************

.. list-table:: skip_mode parameters
   :widths: 20 30 15 25 20
   :header-rows: 1

   * - Name
     - Description
     - Type
     - Available value
     - Required
   * - **type**
     - Skip mode to use
     - string
     - ``fast``, ``complete``, ``custom``
     - Yes
   * - **rules**
     - | Associated rules to activate (``custom``), or to override defaults (``fast``,
       | ``complete``)
     - list
     - See available rules
     - | Yes for ``custom``. Optional for ``fast``
       | and ``complete``

The ``fast`` mode activates every rule that skips calculations
(``incomplete_disparity_range`` and ``score_threshold`` with default parameters).

.. warning::
   **The ``fast`` skip mode can affect results.** Using skip rules means some points may not have a
   disparity estimate. Verify that this trade-off (speed vs completeness) is acceptable
   for your use case.

The ``complete`` mode activates every rule: those that skip calculations,
and those that keep extra information about the computations done in Pandora2D.

For ``fast`` and ``complete``, ``rules`` is optional. It can override the default parameter
values of some rules.
This does not apply to ``incomplete_disparity_range``, which has no parameter.
When ``rules`` is absent, Pandora2D adds the default rules to the output saved configuration.

The ``custom`` mode activates only the rules listed in ``rules``.
The list is required and must not be empty.

A rule type cannot be repeated.
A rule that is not registered is rejected.


Available rules
***************

.. list-table:: Available rules
   :widths: 30 20 50
   :header-rows: 1

   * - Name
     - Category
     - Description
   * - **incomplete_disparity_range**
     - skip
     - | Rule with no parameter. A point is not computed as soon as at least one disparity of its user search range is invalid.
       | Disparities added by refinement margins are ignored by this check.
       | Without this rule, a point is skipped only if every disparity of its cost surface is invalid.
   * - **score_threshold**
     - skip
     - | Rule with parameter **value** (float, default ``0.5``). Requires a **refinement** step in the pipeline.
       | After the disparity step, points whose best correlation score does not meet the threshold do not go through refinement.
       | For correlation metrics where higher is better, refinement is skipped when the score is below the threshold;
       | for cost metrics where lower is better, refinement is skipped when the score is above the threshold.
       | The ``P2D_SCORE_THRESHOLD`` band is added to the validity mask.


**Examples**

.. code:: json

    "skip_mode": { "type": "fast" }

.. code:: json

    "skip_mode": {
      "type": "fast",
      "rules": [
        { "type": "score_threshold", "value": 0.4 }
      ]
    }

.. code:: json

    "skip_mode": { "type": "complete" }

.. code:: json

    "skip_mode": {
      "type": "custom",
      "rules": [
        { "type": "incomplete_disparity_range" },
        { "type": "score_threshold", "value": 0.6 }
      ]
    }

The saved output ``skip_mode`` section lists in ``rules`` every rule applied during the Pandora2D run,
including those that were not given in the input configuration.
