.. _ml-module:

=======================
Machine Learning Module
=======================

.. note::

   **Status: Beta** - This module is actively used by the legacy simulation path.
   Orchestrator integration is planned for a future version.

Quick Summary: What This Module Is and Isn't
============================================

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - This Module IS
     - This Module IS NOT
   * - Utilities for supervised/unsupervised learning
     - Reinforcement learning (that's ``fusion/modules/rl/``)
   * - Feature extraction for traditional ML
     - RL feature extractors (that's ``fusion/modules/rl/feat_extrs/``)
   * - Model save/load, evaluation, visualization
     - RL policies or environments
   * - Used by legacy path (SDNController)
     - Currently used by orchestrator (planned for v6.x)

**If you want RL**, go to :ref:`modules-directory` and look at the RL section.

Overview
========

.. admonition:: At a Glance
   :class: tip

   :Purpose: Utilities for supervised/unsupervised ML in network optimization
   :Location: ``fusion/modules/ml/``
   :Status: **Beta** - Actively used by legacy path
   :Used By: Legacy path (SDNController, SimulationEngine)
   :Planned: Orchestrator integration in v6.x

The ML module provides utilities for traditional machine learning approaches to
network optimization - specifically supervised learning (SL) and unsupervised
learning (USL). It does NOT contain the models themselves; you bring your own
models (sklearn, tensorflow, pytorch) and use these utilities around them.

What This Module Provides
-------------------------

1. **Feature Engineering**: Extract features from network state for ML models
2. **Preprocessing**: Data transformation, normalization, balancing
3. **Model I/O**: Save and load trained models
4. **Evaluation**: Metrics calculation and model comparison
5. **Visualization**: Plotting for ML results (confusion matrices, feature importance)

Relationship with MLControlPolicy
==================================

There are currently **two separate** ML-related areas in FUSION:

.. list-table::
   :header-rows: 1
   :widths: 20 30 30 20

   * - Component
     - Location
     - Purpose
     - Status
   * - ML Utilities (this module)
     - ``fusion/modules/ml/``
     - Feature extraction, preprocessing, model I/O, evaluation
     - Used by legacy path
   * - ML Control Policy
     - ``fusion/policies/ml_policy.py``
     - Path selection using pre-trained models
     - Used by orchestrator

Current State (Not Yet Integrated)
----------------------------------

These two components were built independently and **do not share code**:

- ``fusion/modules/ml/`` has ``extract_ml_features()`` that works with legacy dicts (``engine_props``, ``sdn_props``)
- ``fusion/policies/ml_policy.py`` has its own ``FeatureBuilder`` that works with domain objects (``Request``, ``NetworkState``)
- Both have separate model loading implementations

.. code-block:: text

   Current Architecture (Separate):

   +---------------------------+          +---------------------------+
   | fusion/modules/ml/        |          | fusion/policies/          |
   |                           |          | ml_policy.py              |
   +---------------------------+          +---------------------------+
   | extract_ml_features()     |          | FeatureBuilder            |
   | load_model()              |          | load_model()              |
   | save_model()              |          | MLControlPolicy           |
   | evaluate_classifier()     |          |                           |
   +---------------------------+          +---------------------------+
            |                                        |
            v                                        v
   +---------------------------+          +---------------------------+
   | Legacy Path               |          | Orchestrator Path         |
   | (engine_props, sdn_props) |          | (Request, NetworkState)   |
   +---------------------------+          +---------------------------+

Future Integration (Planned for v6.x)
-------------------------------------

The goal is to unify these so that ``MLControlPolicy`` uses this module's utilities:

- ``MLControlPolicy`` would use this module's feature extraction (adapted for domain objects)
- Model I/O would be consolidated into this module
- Evaluation and visualization utilities would be available for policy analysis

.. code-block:: text

   Future Architecture (Integrated):

   +-----------------------------------------------+
   | fusion/modules/ml/                            |
   +-----------------------------------------------+
   | extract_ml_features()  <-- adapted for both   |
   | load_model()           <-- unified            |
   | save_model()                                  |
   | evaluate_classifier()                         |
   +-----------------------------------------------+
            |                     |
            v                     v
   +------------------+   +------------------+
   | Legacy Path      |   | MLControlPolicy  |
   | SDNController    |   | (orchestrator)   |
   +------------------+   +------------------+

This will eliminate duplicate code and provide a single, well-tested ML utility layer for all paths.

Understanding the Learning Landscape in FUSION
==============================================

FUSION has **two separate modules** for learning-based approaches:

.. code-block:: text

   +===========================================================================+
   |                    LEARNING IN FUSION                                     |
   +===========================================================================+
   |                                                                           |
   |   +---------------------------+     +---------------------------+         |
   |   |   fusion/modules/ml/      |     |   fusion/modules/rl/      |         |
   |   |   (THIS MODULE)           |     |                           |         |
   |   +---------------------------+     +---------------------------+         |
   |   |                           |     |                           |         |
   |   | Supervised Learning (SL)  |     | Reinforcement Learning    |         |
   |   | Unsupervised Learning(USL)|     |                           |         |
   |   |                           |     |                           |         |
   |   | - Feature extraction      |     | - Policies (BC, IQL, etc) |         |
   |   | - Preprocessing           |     | - Environments            |         |
   |   | - Model save/load         |     | - Feature extractors      |         |
   |   | - Evaluation metrics      |     | - SB3 integration         |         |
   |   | - Visualization           |     | - Offline RL              |         |
   |   |                           |     |                           |         |
   |   +---------------------------+     +---------------------------+         |
   |              |                                   |                        |
   |              v                                   v                        |
   |   +---------------------------+     +---------------------------+         |
   |   | LEGACY PATH               |     | ORCHESTRATOR PATH         |         |
   |   | use_orchestrator=False    |     | use_orchestrator=True     |         |
   |   |                           |     |                           |         |
   |   | SDNController calls       |     | RLSimulationAdapter       |         |
   |   | get_ml_obs() for features |     | uses RL policies          |         |
   |   | and loads ML models       |     |                           |         |
   |   |                           |     |                           |         |
   |   | STATUS: WORKS             |     | ML: Planned for v6.x      |         |
   |   +---------------------------+     +---------------------------+         |
   |                                                                           |
   +===========================================================================+

Key Differences
---------------

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * - Aspect
     - ML Module (this)
     - RL Module
   * - Learning Type
     - Supervised / Unsupervised
     - Reinforcement Learning
   * - Training
     - Offline on collected data
     - Online or offline from behavior
   * - Legacy Integration
     - **Works** (deploy_model=True)
     - N/A
   * - Orchestrator Integration
     - Planned for v6.x
     - **Works**
   * - Feature Extraction
     - ``get_ml_obs()`` / ``extract_ml_features()``
     - ``feat_extrs/`` (GNN, Graphormer, etc.)

Current Integration Status
==========================

Visual: How ML Module Integrates with Legacy Path
-------------------------------------------------

.. code-block:: text

   +===========================================================================+
   |                     LEGACY PATH (WORKS)                                    |
   |                  (use_orchestrator=False)                                  |
   |                  (deploy_model=True)                                       |
   +===========================================================================+
                              |
                              v
   +-----------------------------+
   | SimulationEngine            |
   | (fusion/core/simulation.py) |
   |                             |
   | At startup:                 |
   | load_model() ---------------|---> fusion/modules/ml/model_io.py
   |                             |     Loads trained ML model
   +-------------+---------------+
                 |
                 | passes ml_model to
                 v
   +---------------------------------+
   | SDNController                   |
   | (fusion/core/sdn_controller.py) |
   |                                 |
   | During routing:                 |
   | get_ml_obs() -------------------|---> fusion/modules/ml/feature_engineering.py
   |                                 |     Extracts features from network state
   |                                 |
   | ml_model.predict() -------------|---> Your trained model
   |                                 |     Makes routing decision
   +---------------------------------+

   +===========================================================================+
   |                     ORCHESTRATOR PATH                                      |
   |                  (use_orchestrator=True)                                   |
   +===========================================================================+
                              |
                              v
   +---------------------------+
   | SDNOrchestrator           |
   |                           |
   | Currently: Uses RL module |
   | Planned: ML integration   |
   |          in v6.x          |
   +---------------------------+

What Works and What's Planned
-----------------------------

.. list-table::
   :header-rows: 1
   :widths: 40 30 30

   * - Capability
     - Legacy Path
     - Orchestrator Path
   * - Feature extraction (get_ml_obs)
     - **Works**
     - Planned v6.x
   * - Model loading
     - **Works**
     - Planned v6.x
   * - Preprocessing utilities
     - Works
     - Works (standalone)
   * - Evaluation metrics
     - Works
     - Works (standalone)
   * - Visualization
     - Works
     - Works (standalone)

About the Visualization Code
============================

**Q: Why does this module have its own visualization.py?**

FUSION has a central visualization module at ``fusion/visualization/`` with a plugin
architecture. However, the ML module's visualization functions have not yet been
integrated into it.

.. code-block:: text

   Central Module:
   fusion/visualization/           - Central visualization with plugin architecture

   Module-specific (uses plugin system):
   fusion/modules/routing/visualization/   - Routing plugin
   fusion/modules/spectrum/visualization/  - Spectrum plugin
   fusion/modules/snr/visualization/       - SNR plugin
   fusion/modules/rl/visualization/        - RL plugin

   Not yet integrated:
   fusion/modules/ml/visualization.py      - ML plots (to be integrated)

The ML visualization functions (confusion matrices, feature importance, cluster plots)
are planned to be integrated into the central plugin system in a future version.

Module Components
=================

.. code-block:: text

   fusion/modules/ml/
   |-- __init__.py            # Public API exports
   |-- README.md              # Module documentation
   |-- TODO.md                # Development roadmap
   |-- constants.py           # Shared constants
   |-- feature_engineering.py # Feature extraction (get_ml_obs)
   |-- preprocessing.py       # Data transformation
   |-- model_io.py            # Model save/load
   |-- evaluation.py          # Metrics calculation
   |-- visualization.py       # Plotting utilities
   `-- registry.py            # Model registry (currently empty)

feature_engineering.py
----------------------

:Purpose: Extract features from network state for ML models
:Key Function: ``extract_ml_features()`` (alias: ``get_ml_obs()``)
:Used By: SDNController in legacy path

.. code-block:: python

   from fusion.modules.ml import extract_ml_features

   # Extract features for ML model input
   features_df = extract_ml_features(
       request_dict=request,
       engine_properties=engine_props,
       sdn_properties=sdn_props
   )

   # Use with your trained model
   prediction = model.predict(features_df)

model_io.py
-----------

:Purpose: Save and load trained models
:Key Functions: ``save_model()``, ``load_model()``
:Used By: SimulationEngine in legacy path

.. code-block:: python

   from fusion.modules.ml import save_model, load_model

   # Save trained model
   save_model(sim_dict, model, "random_forest", "1000")

   # Load model (SimulationEngine does this automatically when deploy_model=True)
   model = load_model(engine_properties=engine_props)

preprocessing.py
----------------

:Purpose: Data transformation and preparation
:Key Functions: ``process_training_data()``, ``balance_training_data()``, ``normalize_features()``

.. code-block:: python

   from fusion.modules.ml import (
       process_training_data,
       balance_training_data,
       normalize_features
   )

   # Process raw simulation data
   processed = process_training_data(sim_dict, raw_data, erlang)

   # Balance classes (for imbalanced datasets)
   balanced = balance_training_data(processed)

   # Normalize features
   normalized = normalize_features(balanced)

evaluation.py
-------------

:Purpose: Calculate evaluation metrics
:Key Functions: ``evaluate_classifier()``, ``evaluate_regressor()``, ``cross_validate_model()``

.. code-block:: python

   from fusion.modules.ml import evaluate_classifier, compare_models

   # Evaluate a classifier
   metrics = evaluate_classifier(model, test_features, test_labels)
   # Returns: accuracy, precision, recall, f1, confusion_matrix

   # Compare multiple models
   comparison = compare_models([model1, model2], test_data, labels)

visualization.py
----------------

:Purpose: ML-specific plotting
:Key Functions: ``plot_confusion_matrix()``, ``plot_feature_importance()``, ``plot_2d_clusters()``

.. code-block:: python

   from fusion.modules.ml import plot_confusion_matrix, plot_feature_importance

   # Plot confusion matrix
   plot_confusion_matrix(y_true, y_pred, class_names)

   # Plot feature importance
   plot_feature_importance(model, feature_names)

Related: fusion/core/ml_metrics.py
----------------------------------

There's also ``fusion/core/ml_metrics.py`` which collects training data during
simulation. This is related but separate:

- ``MLMetricsCollector`` collects data during simulation runs
- The collected data can then be used to train models
- Those trained models are loaded/used via this ML module

Development Guide
=================

Using the ML Module (Legacy Path)
---------------------------------

1. **Collect training data** during simulations using ``MLMetricsCollector``
2. **Process the data** using ``preprocessing.py`` utilities
3. **Train your model** (sklearn, tensorflow, etc. - not provided)
4. **Save the model** using ``save_model()``
5. **Run simulations** with ``deploy_model=True`` - model is loaded automatically

Testing
=======

.. code-block:: bash

   # Run ML module tests
   pytest fusion/modules/tests/ml/ -v

   # Run with coverage
   pytest fusion/modules/tests/ml/ -v --cov=fusion.modules.ml

Related Documentation
=====================

- :ref:`modules-directory` - Overview of all modules including RL
- :ref:`core-module` - SimulationEngine and SDNController integration
- ``fusion/modules/rl/`` - RL module (used by orchestrator)
- ``fusion/core/ml_metrics.py`` - Training data collection
