===============
Data Management
===============

Guide to managing simulation data, results, training artifacts, and model files in FUSION.

.. contents:: Table of Contents
   :local:
   :depth: 2

Overview
========

FUSION generates and manages several types of data:

- **Simulation Results**: Blocking probability, utilization metrics, performance statistics
- **Training Data**: ML/RL training datasets
- **Model Artifacts**: Trained ML models and RL agents
- **Network Snapshots**: Detailed network state captures
- **Logs**: Execution logs, debug information

Proper data management ensures reproducibility, efficient storage, and easy analysis.

Directory Structure
===================

Default Organization
--------------------

FUSION uses a standard directory structure:

.. code-block:: text

   FUSION/
   ├── data/
   │   ├── results/           # Simulation outputs
   │   │   ├── NSFNet/
   │   │   │   ├── 2025-01-15/
   │   │   │   │   ├── 14-30-00/
   │   │   │   │   │   ├── results.json
   │   │   │   │   │   ├── summary.txt
   │   │   │   │   │   └── snapshots/
   │   │   │   │   └── 15-45-22/
   │   │   │   └── 2025-01-16/
   │   │   └── COST239/
   │   ├── ml_training/       # ML training data
   │   ├── plots/             # Generated visualizations
   │   └── topologies/        # Custom network topologies
   ├── logs/
   │   ├── ml_models/         # Trained ML models
   │   ├── rl_models/         # Trained RL agents
   │   ├── rl_tensorboard/    # TensorBoard logs
   │   └── simulation.log     # Application logs
   └── configs/               # Configuration files

Customizing Paths
-----------------

Override default paths via configuration:

.. code-block:: ini

   [file_settings]
   output_dir = /custom/path/to/results
   log_dir = /custom/path/to/logs

Or environment variables:

.. code-block:: bash

   export FUSION_DATA_DIR=/path/to/data
   export FUSION_LOG_DIR=/path/to/logs

Simulation Results
==================

Result File Formats
-------------------

FUSION supports multiple output formats:

**JSON** (default):
   Human-readable, widely compatible

   .. code-block:: json

      {
        "erlang": 400,
        "blocking_probability": 0.0234,
        "bandwidth_blocking": 12.5,
        "num_requests": 1000,
        "iterations": 5
      }

**CSV**:
   Spreadsheet-compatible, good for analysis

   .. code-block:: csv

      erlang,blocking_probability,bandwidth_blocking
      300,0.001,1.2
      400,0.023,12.5
      500,0.089,45.3

**HDF5**:
   Efficient binary format for large datasets

   .. code-block:: python

      import h5py

      with h5py.File('results.h5', 'r') as f:
          blocking_prob = f['blocking_probability'][:]

Configure format in config file:

.. code-block:: ini

   [file_settings]
   file_type = json  # or csv, hdf5

Result File Contents
--------------------

Standard result files include:

- **Simulation Parameters**: Erlang values, iterations, network info
- **Performance Metrics**:
  - Blocking probability
  - Bandwidth blocking
  - Spectrum utilization
  - Average path length
- **Per-Iteration Statistics**: Individual iteration results
- **Metadata**: Timestamps, configuration hash, version info

Loading Results
---------------

**Python**:

.. code-block:: python

   import json

   with open('data/results/NSFNet/2025-01-15/14-30-00/results.json') as f:
       results = json.load(f)

   blocking_prob = results['blocking_probability']

**Pandas**:

.. code-block:: python

   import pandas as pd

   # CSV
   df = pd.read_csv('results.csv')

   # JSON
   df = pd.read_json('results.json')

   # Query data
   high_load = df[df['erlang'] > 500]

Network Snapshots
=================

Snapshot Configuration
----------------------

Enable snapshots in configuration:

.. code-block:: ini

   [general_settings]
   save_snapshots = True
   snapshot_step = 100      # Save every 100 requests
   save_start_end_slots = True

Snapshot Contents
-----------------

Each snapshot captures:

- **Timestamp**: Request number when snapshot was taken
- **Spectrum State**: Occupancy of all spectrum slots
- **Active Connections**: List of established connections with:
  - Source/destination nodes
  - Path taken
  - Allocated slots
  - Modulation format
- **Link Utilization**: Per-link spectrum usage
- **Network Metrics**: Current blocking probability, utilization

Snapshot Format
---------------

Snapshots are saved as JSON:

.. code-block:: json

   {
     "request_number": 100,
     "timestamp": "2025-01-15T14:30:45",
     "spectrum_state": {
       "(0,1)": [0, 0, 1, 1, 1, 0, 0, ...],
       "(1,2)": [0, 1, 1, 1, 0, 0, 0, ...]
     },
     "active_connections": [
       {
         "source": 0,
         "destination": 5,
         "path": [0, 1, 3, 5],
         "slots": [10, 15],
         "modulation": "16QAM"
       }
     ]
   }

Using Snapshots
---------------

**Load and analyze**:

.. code-block:: python

   import json

   with open('data/results/.../snapshots/snapshot_100.json') as f:
       snapshot = json.load(f)

   # Analyze spectrum utilization
   for link, slots in snapshot['spectrum_state'].items():
       utilization = sum(slots) / len(slots)
       print(f"Link {link}: {utilization:.2%} utilized")

**Visualize**:

.. code-block:: python

   from fusion.visualization.spectrum_viz import plot_spectrum_heatmap

   plot_spectrum_heatmap(
       network_spectrum_dict=snapshot['spectrum_state'],
       save_path='plots/spectrum_snapshot.png'
   )

ML/RL Training Data
===================

Training Data Collection
------------------------

Enable ML data collection:

.. code-block:: ini

   [ml_settings]
   collect_training_data = True
   training_data_output = data/ml_training/

Data is saved as CSV after simulation completes.

Training Data Format
--------------------

Standard CSV format:

.. code-block:: text

   old_bandwidth,path_length,longest_reach,average_congestion,num_segments
   100.0,450.2,1200.0,0.35,1
   200.0,850.7,1200.0,0.67,2
   400.0,1150.3,800.0,0.82,4

Use with scikit-learn:

.. code-block:: python

   import pandas as pd
   from sklearn.model_selection import train_test_split

   # Load data
   data = pd.read_csv('data/ml_training/training_data.csv')

   # Split features and labels
   X = data.drop('num_segments', axis=1)
   y = data['num_segments']

   # Train/test split
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

Model Artifacts
===============

ML Model Storage
----------------

Models saved with metadata:

.. code-block:: text

   logs/
   └── ml_models/
       ├── random_forest_500.joblib      # Model file
       ├── random_forest_500_metadata.json  # Training info
       └── feature_names.json            # Feature information

**Load models**:

.. code-block:: python

   import joblib

   model = joblib.load('logs/ml_models/random_forest_500.joblib')

   # Check metadata
   with open('logs/ml_models/random_forest_500_metadata.json') as f:
       metadata = json.load(f)
       print(f"Accuracy: {metadata['accuracy']}")

RL Agent Storage
----------------

RL agents saved as ZIP archives:

.. code-block:: text

   logs/
   └── rl_models/
       ├── ppo_agent_best.zip       # Best agent
       ├── ppo_agent_final.zip      # Final agent
       └── checkpoints/
           ├── ppo_10000.zip
           ├── ppo_20000.zip
           └── ppo_30000.zip

**Load agents**:

.. code-block:: python

   from stable_baselines3 import PPO

   agent = PPO.load('logs/rl_models/ppo_agent_best.zip')

TensorBoard Logs
----------------

RL training logs for TensorBoard:

.. code-block:: bash

   tensorboard --logdir logs/rl_tensorboard/

Logs include:
   - Episode rewards
   - Loss curves
   - Policy entropy
   - Learning rate schedule

Data Versioning
===============

Version Control
---------------

Track experiments with version metadata:

.. code-block:: python

   metadata = {
       'version': '1.0.0',
       'config_hash': 'abc123',
       'git_commit': 'def456',
       'timestamp': '2025-01-15T14:30:00',
       'parameters': {...}
   }

Save with results:

.. code-block:: python

   import json

   with open('results_metadata.json', 'w') as f:
       json.dump(metadata, f, indent=2)

Git Integration
---------------

Track configuration files and code:

.. code-block:: bash

   git add configs/experiment.ini
   git commit -m "Add experiment configuration"
   git tag v1.0-experiment

DVC (Data Version Control)
---------------------------

For large datasets, use DVC:

.. code-block:: bash

   # Initialize DVC
   dvc init

   # Track data
   dvc add data/results/large_dataset.h5

   # Commit DVC metadata
   git add data/results/large_dataset.h5.dvc .gitignore
   git commit -m "Add large dataset"

Storage Management
==================

Disk Usage Monitoring
---------------------

Check FUSION data usage:

.. code-block:: bash

   du -sh data/
   du -sh logs/

**Breakdown by directory**:

.. code-block:: bash

   du -h --max-depth=2 data/

Cleanup Strategies
------------------

**Remove old simulations**:

.. code-block:: bash

   # Keep only last 7 days
   find data/results -type d -mtime +7 -exec rm -rf {} +

**Compress archives**:

.. code-block:: bash

   # Compress old results
   tar -czf results_2024.tar.gz data/results/2024-*
   rm -rf data/results/2024-*

**Remove intermediate snapshots**:

.. code-block:: ini

   [general_settings]
   save_snapshots = False  # Disable if not needed

**Clean TensorBoard logs**:

.. code-block:: bash

   # Remove training runs older than 30 days
   find logs/rl_tensorboard -type d -mtime +30 -exec rm -rf {} +

Data Export
===========

Export for Analysis
-------------------

**Export to CSV for Excel/R**:

.. code-block:: python

   import pandas as pd

   # Convert JSON to CSV
   df = pd.read_json('results.json')
   df.to_csv('results.csv', index=False)

**Export to MATLAB**:

.. code-block:: python

   from scipy.io import savemat

   savemat('results.mat', {'blocking_prob': blocking_prob_array})

**Export to LaTeX**:

.. code-block:: python

   import pandas as pd

   df = pd.read_csv('results.csv')
   latex_table = df.to_latex(index=False)

   with open('table.tex', 'w') as f:
       f.write(latex_table)

Sharing Results
---------------

**Create archive for sharing**:

.. code-block:: bash

   tar -czf experiment_results.tar.gz \
       data/results/experiment/ \
       configs/experiment.ini \
       plots/

**Include reproduction info**:

.. code-block:: bash

   echo "FUSION v1.0.0 - Experiment Results" > README.txt
   echo "Config: configs/experiment.ini" >> README.txt
   echo "Ran on: $(date)" >> README.txt

Data Backup
===========

Backup Strategies
-----------------

**Local backup**:

.. code-block:: bash

   # Automated daily backup
   tar -czf backup_$(date +%Y%m%d).tar.gz data/ logs/

**Cloud backup (AWS S3)**:

.. code-block:: bash

   aws s3 sync data/results s3://mybucket/fusion-results/

**Rsync to remote server**:

.. code-block:: bash

   rsync -avz data/ user@server:/backup/fusion/data/

Backup Schedule
---------------

Recommended backup frequency:

- **Simulation results**: After each major experiment
- **Trained models**: Immediately after training
- **Configuration files**: Version control (Git)
- **Raw data**: Weekly/monthly

Best Practices
==============

Organization
------------

1. **Use descriptive names**: ``nsfnet_rl_ppo_2025-01-15`` not ``sim1``
2. **Include timestamps**: Automatic via FUSION's directory structure
3. **Document experiments**: Add README files to result directories
4. **Tag important runs**: Use version tags or special markers

Reproducibility
---------------

1. **Save configurations**: Always store the INI file with results
2. **Log environment**: Record Python version, dependencies
3. **Version code**: Use Git tags for experiments
4. **Document changes**: Maintain a changelog

Storage Efficiency
------------------

1. **Compress old data**: Use gzip/tar for archival
2. **Selective snapshots**: Only save when needed
3. **Prune checkpoints**: Keep only best/final models
4. **Use HDF5**: For large numerical datasets

Common Issues
=============

Disk Space Full
---------------

**Error**: ``OSError: [Errno 28] No space left on device``

**Solutions**:
- Clean old results: ``rm -rf data/results/old_experiments``
- Compress archives: ``tar -czf old_data.tar.gz data/old/``
- Disable snapshots if not needed

Permission Errors
-----------------

**Error**: ``PermissionError: [Errno 13] Permission denied``

**Solution**:

.. code-block:: bash

   chmod -R u+w data/
   chmod -R u+w logs/

Corrupted Files
---------------

**Error**: ``JSONDecodeError`` or ``EOFError`` loading files

**Solutions**:
- Check file size: ``ls -lh results.json``
- Validate JSON: ``python -m json.tool results.json``
- Restore from backup if corrupted

Next Steps
==========

- :doc:`running_simulations` - Generate simulation data
- :doc:`visualization` - Analyze and plot results
- :doc:`machine_learning` - Manage ML training data
- :doc:`reinforcement_learning` - Manage RL artifacts

See Also
========

* :doc:`cli_reference` - Command-line data management
* :doc:`configuration_reference` - Output configuration options
* :doc:`../developer/contributing` - Data handling in development
