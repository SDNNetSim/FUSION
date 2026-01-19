==================
Machine Learning
==================

This guide covers integrating machine learning algorithms into FUSION for intelligent routing, spectrum allocation, and network optimization decisions.

.. contents:: Table of Contents
   :local:
   :depth: 2

.. note::

   Machine learning capabilities in FUSION are under active development. This guide covers current functionality and planned features.

Overview
========

FUSION supports machine learning models for:

- **Routing decisions**: ML algorithms select optimal paths based on network state
- **Spectrum allocation**: Intelligent slot assignment to minimize fragmentation
- **Modulation format selection**: Data-driven modulation choices
- **Network optimization**: Predict and prevent congestion

ML models learn from simulation data to make better decisions than traditional heuristics.

Benefits of ML in Optical Networks
-----------------------------------

**Improved Performance**:
   ML models can achieve 10-30% lower blocking probability compared to first-fit/k-shortest path

**Adaptability**:
   Models learn from network-specific patterns and traffic characteristics

**Multi-Objective Optimization**:
   Balance blocking probability, spectrum utilization, and path length simultaneously

**Scalability**:
   Once trained, inference is fast and suitable for real-time decision-making

Getting Started with ML
=======================

Basic ML Workflow
-----------------

1. **Collect Training Data**: Run simulations with data collection enabled
2. **Train Models**: Use collected data to train ML algorithms
3. **Deploy Models**: Load trained models into production simulations
4. **Evaluate Performance**: Compare ML vs baseline algorithms

Quick Start Example
-------------------

**Step 1: Collect Training Data**

.. code-block:: ini

   [general_settings]
   erlang_start = 400
   erlang_stop = 600
   num_requests = 5000

   [ml_settings]
   deploy_model = False
   collect_training_data = True

.. code-block:: bash

   fusion-sim --config collect_data.ini

**Step 2: Train ML Model**

.. code-block:: bash

   fusion-train --agent_type ml \
                --config training_config.ini \
                --algorithm decision_tree

**Step 3: Deploy and Evaluate**

.. code-block:: ini

   [ml_settings]
   deploy_model = True
   model_path = logs/decision_tree/decision_tree_500.joblib

.. code-block:: bash

   fusion-sim --config deploy_ml.ini

Training Data Collection
=========================

What Data is Collected
----------------------

FUSION's ``MLMetricsCollector`` captures features for each connection request:

**Request Features**:
   - Requested bandwidth (Gbps)
   - Source and destination nodes
   - Modulation format compatibility

**Network State Features**:
   - Average link congestion on candidate paths
   - Path length (km)
   - Spectrum fragmentation metrics
   - Per-core utilization (for multi-core fibers)

**Target Variables**:
   - Number of transponder segments needed
   - Blocking outcome (blocked/successful)
   - Spectrum slots allocated

Enabling Data Collection
------------------------

Set ``collect_training_data = True`` in your configuration:

.. code-block:: ini

   [ml_settings]
   deploy_model = False
   collect_training_data = True
   training_data_output = data/ml_training/

Run simulations normally - training data is saved automatically:

.. code-block:: bash

   fusion-sim --config data_collection.ini

Data Format
-----------

Training data is saved as CSV with the following structure:

.. code-block:: text

   old_bandwidth,path_length,longest_reach,average_congestion,num_segments
   100.0,450.2,1200.0,0.35,1
   200.0,850.7,1200.0,0.67,2
   400.0,1150.3,800.0,0.82,4

This can be loaded directly into scikit-learn, pandas, or other ML frameworks.

Training ML Models
==================

Supported Algorithms
--------------------

FUSION supports scikit-learn compatible models:

**Decision Trees**:
   Fast, interpretable, good baseline performance

.. code-block:: bash

   fusion-train --agent_type ml --algorithm decision_tree

**Random Forests**:
   Ensemble method, robust to overfitting

.. code-block:: bash

   fusion-train --agent_type ml --algorithm random_forest

**Gradient Boosting**:
   State-of-the-art performance, slower training

.. code-block:: bash

   fusion-train --agent_type ml --algorithm gradient_boosting --n_estimators 200

**Support Vector Machines**:
   Good for high-dimensional feature spaces

.. code-block:: bash

   fusion-train --agent_type ml --algorithm svm

**Custom Models**:
   Any scikit-learn compatible estimator

Training Configuration
----------------------

Configure training parameters in your config file:

.. code-block:: ini

   [ml_settings]
   # Training mode
   deploy_model = False
   collect_training_data = False

   # Data paths
   training_data_path = data/ml_training/nsfnet_training_data.csv
   model_output_dir = logs/ml_models/

   # Algorithm selection
   algorithm = random_forest

   # Hyperparameters
   n_estimators = 100
   max_depth = 20
   min_samples_split = 10
   random_state = 42

   # Training options
   test_size = 0.2
   cv_folds = 5

Training from Python
--------------------

For advanced workflows, train models programmatically:

.. code-block:: python

   import pandas as pd
   from sklearn.ensemble import RandomForestClassifier
   from sklearn.model_selection import train_test_split
   from fusion.modules.ml.evaluation import evaluate_classifier
   from fusion.modules.ml.model_io import save_model

   # Load training data
   data = pd.read_csv('data/ml_training/training_data.csv')
   X = data[['old_bandwidth', 'path_length', 'average_congestion']]
   y = data['num_segments']

   # Split data
   X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.2, random_state=42
   )

   # Train model
   model = RandomForestClassifier(n_estimators=100, max_depth=20)
   model.fit(X_train, y_train)

   # Evaluate
   predictions = model.predict(X_test)
   metrics = evaluate_classifier(y_test, predictions)
   print(f"Accuracy: {metrics['accuracy']:.2%}")

   # Save model
   save_model(
       simulation_dict={'train_file_path': 'nsfnet'},
       model=model,
       algorithm='random_forest',
       erlang='500',
       metadata=metrics
   )

Model Evaluation
================

Evaluation Metrics
------------------

FUSION provides comprehensive evaluation utilities:

**Classification Metrics** (for discrete predictions):

.. code-block:: python

   from fusion.modules.ml.evaluation import evaluate_classifier

   metrics = evaluate_classifier(y_true, y_pred, class_names=['1', '2', '3', '4'])

   print(f"Accuracy: {metrics['accuracy']:.2%}")
   print(f"F1-Score: {metrics['f1_score']:.3f}")
   print(f"Confusion Matrix:\n{metrics['confusion_matrix']}")

Returns:
   - Accuracy, precision, recall, F1-score
   - Per-class metrics
   - Confusion matrix
   - ROC-AUC (for binary classification)

**Regression Metrics** (for continuous predictions):

.. code-block:: python

   from fusion.modules.ml.evaluation import evaluate_regressor

   metrics = evaluate_regressor(y_true, y_pred)

   print(f"RMSE: {metrics['rmse']:.4f}")
   print(f"R²: {metrics['r2']:.3f}")
   print(f"MAE: {metrics['mae']:.4f}")

Returns:
   - MSE, RMSE, MAE
   - R² score
   - MAPE (Mean Absolute Percentage Error)
   - Max error

Cross-Validation
----------------

Evaluate model robustness with cross-validation:

.. code-block:: python

   from fusion.modules.ml.evaluation import cross_validate_model

   cv_results = cross_validate_model(
       model=model,
       X=X_train,
       y=y_train,
       cv_folds=5,
       scoring=['accuracy', 'f1_weighted', 'precision_weighted']
   )

   print(f"CV Accuracy: {cv_results['accuracy_mean']:.2%} ± {cv_results['accuracy_std']:.2%}")

Deploying ML Models
===================

Loading Trained Models
----------------------

Enable ML deployment in your configuration:

.. code-block:: ini

   [ml_settings]
   deploy_model = True
   model_path = logs/random_forest/random_forest_500.joblib

   [general_settings]
   route_method = ml  # or allocation_method = ml

FUSION automatically loads the model at simulation start.

Model File Formats
------------------

FUSION supports multiple model formats:

**Joblib** (recommended):
   Native Python serialization, fast loading

   .. code-block:: python

      import joblib
      model = joblib.load('model.joblib')

**Pickle**:
   Standard Python serialization

   .. code-block:: python

      import pickle
      with open('model.pkl', 'rb') as f:
          model = pickle.load(f)

**ONNX** (experimental):
   Cross-platform format, requires ``skl2onnx``

**PMML** (experimental):
   XML-based format, requires ``sklearn2pmml``

Runtime Inference
-----------------

During simulation, the ML model is invoked for each decision:

.. code-block:: python

   from fusion.modules.ml.model_io import load_model

   # Load model
   model = load_model(engine_properties)

   # Extract features from network state
   features = [
       request_bandwidth,
       path_length,
       average_congestion,
       spectrum_fragmentation
   ]

   # Make prediction
   prediction = model.predict([features])[0]

   # Use prediction for routing/allocation decision
   num_segments = int(prediction)

Feature Engineering
===================

Custom Features
---------------

Extend the feature set for better model performance:

.. code-block:: python

   from fusion.modules.ml.feature_engineering import FeatureEngineer

   engineer = FeatureEngineer()

   # Add derived features
   features = engineer.add_features(
       request_dict=request_info,
       network_state=network_spectrum_dict,
       topology=topology
   )

   # Features now include:
   # - Bandwidth/path_length ratio
   # - Normalized congestion
   # - Spectral efficiency metrics
   # - Topology-specific features

Built-in Feature Transformations
---------------------------------

**Normalization**:
   Scale features to [0, 1] range

**Standardization**:
   Zero mean, unit variance

**Polynomial Features**:
   Create interaction terms

**PCA**:
   Dimensionality reduction

.. code-block:: python

   from fusion.modules.ml.preprocessing import normalize_features

   X_normalized = normalize_features(X, method='minmax')

Advanced ML Techniques
======================

Hyperparameter Tuning
---------------------

Optimize model performance with grid search:

.. code-block:: python

   from sklearn.model_selection import GridSearchCV
   from sklearn.ensemble import RandomForestClassifier

   param_grid = {
       'n_estimators': [50, 100, 200],
       'max_depth': [10, 20, None],
       'min_samples_split': [2, 5, 10]
   }

   grid_search = GridSearchCV(
       RandomForestClassifier(),
       param_grid,
       cv=5,
       scoring='f1_weighted'
   )

   grid_search.fit(X_train, y_train)
   best_model = grid_search.best_estimator_

Ensemble Methods
----------------

Combine multiple models for better performance:

.. code-block:: python

   from sklearn.ensemble import VotingClassifier

   ensemble = VotingClassifier(
       estimators=[
           ('rf', random_forest_model),
           ('gb', gradient_boosting_model),
           ('svm', svm_model)
       ],
       voting='soft'  # Use probability averaging
   )

   ensemble.fit(X_train, y_train)

Online Learning
---------------

Update models with new simulation data:

.. code-block:: python

   from sklearn.linear_model import SGDClassifier

   # Initialize with partial_fit capability
   model = SGDClassifier()

   # Train incrementally as new data arrives
   for batch in data_batches:
       model.partial_fit(batch_X, batch_y, classes=np.unique(y))

Model Visualization
===================

Feature Importance
------------------

Understand which features drive model decisions:

.. code-block:: python

   from fusion.modules.ml.visualization import plot_feature_importance

   plot_feature_importance(
       model=model,
       feature_names=['bandwidth', 'path_length', 'congestion'],
       save_path='plots/feature_importance.png'
   )

Decision Tree Visualization
---------------------------

Visualize decision tree structure:

.. code-block:: python

   from fusion.modules.ml.visualization import plot_decision_tree

   plot_decision_tree(
       model=decision_tree_model,
       feature_names=feature_names,
       class_names=class_names,
       save_path='plots/tree_structure.png'
   )

Learning Curves
---------------

Diagnose overfitting/underfitting:

.. code-block:: python

   from fusion.modules.ml.visualization import plot_learning_curve

   plot_learning_curve(
       model=model,
       X=X_train,
       y=y_train,
       cv=5,
       save_path='plots/learning_curve.png'
   )

Best Practices
==============

Data Quality
------------

1. **Sufficient Data**: Collect 10,000+ samples for robust training
2. **Balanced Classes**: Ensure all outcomes are well-represented
3. **Representative Traffic**: Use realistic traffic patterns
4. **Multiple Topologies**: Train on diverse network configurations

Model Selection
---------------

1. **Start Simple**: Begin with decision trees for interpretability
2. **Baseline Comparison**: Always compare against k-shortest + first-fit
3. **Cross-Validation**: Validate on held-out data
4. **Simulation Testing**: Evaluate in full simulations, not just offline metrics

Production Deployment
---------------------

1. **Version Control**: Track model versions with metadata
2. **A/B Testing**: Compare new models against production baselines
3. **Monitoring**: Log prediction distributions and performance
4. **Fallback Strategy**: Revert to heuristics if model fails

Common Issues
=============

Model Won't Load
----------------

**Error**: ``FileNotFoundError: model.joblib not found``

**Solution**:
- Verify ``model_path`` in configuration
- Check model was saved successfully
- Use absolute paths or paths relative to project root

Poor Model Performance
----------------------

**Low accuracy despite good training metrics:**

- **Overfitting**: Reduce model complexity, increase regularization
- **Data leakage**: Ensure test data is truly held-out
- **Distribution shift**: Training and deployment conditions differ

**Model performs worse than baseline:**

- Collect more diverse training data
- Engineer better features
- Try different algorithms
- Verify feature normalization is consistent

Slow Inference
--------------

**Simulation runs slowly with ML:**

- Use simpler models (decision trees vs deep ensembles)
- Pre-compute features when possible
- Enable model caching
- Profile inference time

Next Steps
==========

- :doc:`reinforcement_learning` - Advanced RL-based optimization
- :doc:`../examples/ml_decision_tree` - Complete ML example walkthrough
- :doc:`../concepts/machine_learning_optical` - ML theory for optical networks
- :doc:`visualization` - Visualize ML model performance

See Also
========

* :doc:`running_simulations` - Run simulations with ML models
* :doc:`data_management` - Manage training data and model artifacts
* :doc:`../developer/extending` - Implement custom ML algorithms
* :doc:`../examples/batch_simulations` - Train models on multiple scenarios