========================
Machine Learning Example
========================

Use machine learning to optimize routing and spectrum assignment decisions.

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
========

This example demonstrates using scikit-learn decision tree classifiers for
intelligent network resource allocation based on historical simulation data.

Quick Start
===========

.. code-block:: bash

   # Generate training data
   fusion-sim --config base_config.ini --save_ml_data

   # Train ML model
   fusion-train-ml --algorithm decision_tree --data output/ml_data.csv

   # Run with ML model
   fusion-sim --config base_config.ini --ml_model trained_model.pkl

Complete Workflow
=================

Step 1: Generate Training Data
-------------------------------

Configure data collection:

.. code-block:: ini

   [general_settings]
   save_ml_data = True
   ml_features = path_length,spectrum_utilization,hop_count

   [ml_settings]
   output_data_file = training_data.csv

Step 2: Train Decision Tree Model
----------------------------------

.. code-block:: python

   from fusion.modules.ml.preprocessing import prepare_ml_data
   from fusion.modules.ml.model_io import save_model
   from sklearn.tree import DecisionTreeClassifier

   # Load and preprocess data
   X_train, X_test, y_train, y_test = prepare_ml_data('training_data.csv')

   # Train model
   model = DecisionTreeClassifier(max_depth=10, min_samples_split=20)
   model.fit(X_train, y_train)

   # Evaluate
   accuracy = model.score(X_test, y_test)
   print(f"Model accuracy: {accuracy:.2%}")

   # Save model
   save_model(model, 'decision_tree_model.pkl')

Step 3: Use ML Model in Simulation
-----------------------------------

.. code-block:: ini

   [ml_settings]
   ml_enabled = True
   ml_model_path = decision_tree_model.pkl
   ml_algorithm = decision_tree

See Also
========

* :doc:`rl_training` - Reinforcement learning alternative
* :doc:`../user_guide/machine_learning` - Complete ML guide
* :doc:`../api/modules` - ML module reference
