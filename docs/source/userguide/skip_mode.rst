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
   **Skip mode is still in development.** Currently, only the ``incomplete_disparity_range``
   skip rule is implemented. Other rules like ``score_threshold`` will be added
   in future releases. Additional rules may have restrictions or be adjusted as implementation progresses.


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
     - Associated rules to activate (``custom``), or to override defaults (``fast``, ``complete``)
     - list
     - See available rules
     - Yes for ``custom``. Optional for ``fast`` and ``complete``

The ``fast`` mode activates every rule that skips calculations.

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
     - Skip rule with no parameter.


**Examples**

.. code:: json

    "skip_mode": { "type": "fast" }

.. code:: json

    "skip_mode": {
      "type": "fast",
      "rules": [
        { "type": "incomplete_disparity_range" }
      ]
    }

.. code:: json

    "skip_mode": { "type": "complete" }

.. code:: json

    "skip_mode": {
      "type": "custom",
      "rules": [
        { "type": "incomplete_disparity_range" }
      ]
    }

The saved output ``skip_mode`` section lists in ``rules`` every rule applied during the Pandora2D run,
including those that were not given in the input configuration.
