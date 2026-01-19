.. _unity-module:

============
Unity Module
============

Overview
========

.. admonition:: At a Glance
   :class: tip

   :Purpose: Run FUSION simulations on SLURM-managed HPC clusters
   :Location: ``fusion/unity/``
   :Key Files: ``make_manifest.py``, ``submit_manifest.py``, ``fetch_results.py``
   :Cluster: Any SLURM cluster (named "Unity" after UMass Amherst's cluster)

The ``unity`` module provides a complete workflow for running FUSION simulations
on High-Performance Computing (HPC) clusters that use the **SLURM** workload manager.

.. important::

   **This module is for SLURM clusters only.**

   SLURM (Simple Linux Utility for Resource Management) is the job scheduler used
   by many HPC clusters including Unity at UMass, NERSC, and most university clusters.
   If your cluster uses a different scheduler (PBS, SGE, etc.), this module will
   not work directly.

**What this module does:**

1. **Generate job manifests** - Convert parameter specifications into CSV manifests
2. **Submit to SLURM** - Submit manifests as array jobs to the cluster
3. **Fetch results** - Automatically find and download results to your local machine

Quick Start: End-to-End Example
===============================

This section walks you through the complete workflow from setting up your
environment to downloading results.

Step 1: Create the Virtual Environment on the Cluster
------------------------------------------------------

First, SSH into your cluster and create a virtual environment for FUSION.

.. code-block:: bash

   # SSH into your cluster
   ssh username@unity.rc.umass.edu

   # Navigate to your work directory
   cd /work/username

   # Clone FUSION (if not already done)
   git clone https://github.com/your-org/FUSION.git
   cd FUSION

   # Create virtual environment using the provided script
   ./fusion/unity/scripts/make_unity_venv.sh /work/username/fusion_venv python3.11

   # Activate the virtual environment
   source /work/username/fusion_venv/venv/bin/activate

   # Install FUSION and dependencies
   pip install -e .
   pip install -r requirements.txt

.. note::

   We call it "unity_venv" because our cluster is named Unity, but this works
   on any SLURM cluster. Name it whatever makes sense for your environment.

Step 2: Create a Specification File
-----------------------------------

Create a YAML file that defines your experiment parameters. The module will
automatically expand parameter combinations into individual jobs.

Create ``specs/my_experiment.yaml``:

.. code-block:: yaml

   # Resource allocation for SLURM
   resources:
     partition: gpu-long      # SLURM partition name
     time: "24:00:00"         # Wall clock time (HH:MM:SS)
     mem: "32G"               # Memory per job
     cpus: 8                  # CPU cores per job
     gpus: 1                  # GPUs per job (0 for CPU-only)
     nodes: 1                 # Nodes per job

   # Parameter grid - all combinations will be generated
   grid:
     # Common parameters (same for all jobs)
     common:
       network: "NSFNet"
       num_requests: 10000
       holding_time: 5000
       guard_slots: 1
       cores_per_link: 7
       allocation_method: "first_fit"

     # Parameters to sweep (creates Cartesian product)
     path_algorithm: ["k_shortest_path", "least_congested"]
     erlang_start: [100, 200, 300]
     k_paths: [3, 5]

This specification creates **12 jobs** (2 algorithms x 3 traffic loads x 2 k values).

Step 3: Generate the Manifest
-----------------------------

Run the manifest generator to create job files:

.. code-block:: bash

   # From your FUSION directory
   python -m fusion.unity.make_manifest my_experiment

   # Or with full path
   python -m fusion.unity.make_manifest specs/my_experiment.yaml

**Output structure created:**

.. code-block:: text

   experiments/
     0119/                          # Date (MMDD)
       1430/                         # Time (HHMM)
         NSFNet/                      # Network name
           manifest.csv              # Job parameters (one row per job)
           manifest_meta.json        # Metadata about the manifest

**manifest.csv contents:**

.. code-block:: text

   path_algorithm,erlang_start,erlang_stop,k_paths,network,num_requests,...
   k_shortest_path,100,150,3,NSFNet,10000,...
   k_shortest_path,100,150,5,NSFNet,10000,...
   k_shortest_path,200,250,3,NSFNet,10000,...
   ...

Step 4: Submit Jobs to SLURM
----------------------------

Submit your manifest as a SLURM array job:

.. code-block:: bash

   python -m fusion.unity.submit_manifest \
       experiments/0119/1430/NSFNet \
       run_sim.sh

**What happens:**

1. Reads the manifest CSV
2. Creates a ``jobs/`` directory for SLURM output logs
3. Submits an array job where each task processes one manifest row
4. Returns the SLURM job ID

**Example output:**

.. code-block:: text

   Submitted batch job 12345678

**SLURM logs are saved to:**

.. code-block:: text

   experiments/0119/1430/NSFNet/jobs/
     slurm_12345678_0.out     # First job
     slurm_12345678_1.out     # Second job
     ...

Step 5: Monitor Your Jobs
-------------------------

Use standard SLURM commands to monitor progress:

.. code-block:: bash

   # Check job status
   squeue -u $USER

   # Check detailed job info
   sacct -j 12345678

   # View job output in real-time
   tail -f experiments/0119/1430/NSFNet/jobs/slurm_12345678_0.out

   # Check cluster priority (using provided script)
   ./fusion/unity/scripts/priority.sh

Step 6: Fetch Results to Your Local Machine
-------------------------------------------

Once jobs complete, download results to your local machine.

**On your local machine**, create ``configs/config.yml``:

.. code-block:: yaml

   # Remote paths on the cluster
   metadata_root: "username@unity.rc.umass.edu:/work/username/FUSION/experiments"
   data_root: "username@unity.rc.umass.edu:/work/username/FUSION/data"
   logs_root: "username@unity.rc.umass.edu:/work/username/FUSION/logs"

   # Local destination
   dest: "~/cluster_results"

   # Which experiment to fetch
   experiment: "0119/1430/NSFNet"

   # Set to true to preview without downloading
   dry_run: false

**Run the fetch command:**

.. code-block:: bash

   python -m fusion.unity.fetch_results

**What happens:**

1. Downloads the runs index file from the cluster
2. Identifies all completed simulation outputs
3. Uses rsync to download output data, input configs, and logs
4. Organizes everything in your local destination directory

**Local result structure:**

.. code-block:: text

   ~/cluster_results/
     data/
       NSFNet/
         0119/1430/
           output/
             s1/                  # Seed 1 results
             s2/                  # Seed 2 results
           input/
             sim_input_s1.json
             sim_input_s2.json
     logs/
       k_shortest_path/
         NSFNet/
           0119/1430/
             simulation.log

How It Works
============

Architecture Overview
---------------------

.. code-block:: text

   LOCAL MACHINE                         CLUSTER (SLURM)
   =============                         ===============

   specs/experiment.yaml
          |
          v
   +------------------+
   | make_manifest.py |  ------>  experiments/MMDD/HHMM/network/
   +------------------+              manifest.csv
                                     manifest_meta.json
                                            |
                                            v
                                  +--------------------+
                                  | submit_manifest.py |
                                  +--------------------+
                                            |
                                            v
                                     SLURM Array Job
                                     (sbatch --array=0-N)
                                            |
                                     +------+------+
                                     |      |      |
                                     v      v      v
                                   Job 0  Job 1  Job N
                                     |      |      |
                                     v      v      v
                                  data/output/network/...
                                            |
   +------------------+                     |
   | fetch_results.py |  <------ rsync -----+
   +------------------+
          |
          v
   ~/cluster_results/

Manifest Generation Details
---------------------------

The ``make_manifest.py`` module converts specifications into job manifests.

**Input Modes:**

1. **Grid Mode** (``grid`` or ``grids``): Cartesian product of parameter lists
2. **Explicit Mode** (``jobs``): Manually specified job list

**Grid Expansion Example:**

.. code-block:: yaml

   grid:
     path_algorithm: ["ppo", "dqn"]
     erlang_start: [100, 200]
     k_paths: [3]

Expands to 4 jobs (2 x 2 x 1):

.. code-block:: text

   ppo,  100, 3
   ppo,  200, 3
   dqn,  100, 3
   dqn,  200, 3

**Automatic Erlang Stop:**

If ``erlang_stop`` is not specified, it's automatically set to ``erlang_start + 50``.

**Type Casting:**

All parameters are automatically cast to the correct types based on FUSION's
configuration schema. Booleans, lists, and dicts are properly encoded.

SLURM Submission Details
------------------------

The ``submit_manifest.py`` module submits manifests as SLURM array jobs.

**Environment Variables Passed to Jobs:**

.. code-block:: text

   MANIFEST=/path/to/manifest.csv    # Full path to manifest
   N_JOBS=11                          # Number of jobs (0-indexed)
   JOB_DIR=experiments/0119/1430/net  # Experiment directory
   NETWORK=NSFNet                     # Network name
   DATE=0119                          # Date portion
   JOB_NAME=ppo_100_0119_1430_net     # SLURM job name
   PARTITION=gpu-long                 # SLURM partition
   TIME=24:00:00                      # Time limit
   MEM=32G                            # Memory
   CPUS=8                             # CPUs
   GPUS=1                             # GPUs
   NODES=1                            # Nodes

**Your bash script** (e.g., ``run_sim.sh``) reads these variables and the
``SLURM_ARRAY_TASK_ID`` to determine which manifest row to process.

Result Fetching Details
-----------------------

The ``fetch_results.py`` module uses rsync to download results.

**What Gets Downloaded:**

1. **Output data**: Simulation results (``data/output/...``)
2. **Input configs**: The configuration used for each run (``data/input/...``)
3. **Logs**: Simulation logs organized by algorithm and topology

**Path Conversion:**

The module automatically converts output paths to input paths:

.. code-block:: text

   /work/data/output/NSFNet/exp1/s1  ->  /work/data/input/NSFNet/exp1

**Rsync Options:**

- ``-a``: Archive mode (preserves permissions, timestamps)
- ``-v``: Verbose output
- ``-P``: Show progress and allow resume
- ``--compress``: Compress during transfer

A 3-second delay is added between rsync commands to avoid overwhelming the cluster.

Input Format Reference
======================

Specification File Structure
----------------------------

.. code-block:: yaml

   # REQUIRED: Resource allocation
   resources:
     partition: "gpu"          # SLURM partition
     time: "24:00:00"          # Wall time (HH:MM:SS)
     mem: "32G"                # Memory
     cpus: 8                   # CPU cores
     gpus: 1                   # GPUs (use 0 for CPU-only)
     nodes: 1                  # Nodes

   # OPTION A: Grid-based parameter sweep
   grid:
     common:
       # Parameters that are the same for all jobs
       network: "NSFNet"
       num_requests: 10000

     # Parameters to sweep (lists create combinations)
     path_algorithm: ["ppo", "dqn"]
     erlang_start: [100, 200]

   # OPTION B: Multiple grids
   grids:
     - common:
         network: "NSFNet"
       path_algorithm: ["ppo"]
       erlang_start: [100, 200]
     - common:
         network: "COST239"
       path_algorithm: ["dqn"]
       erlang_start: [300]

   # OPTION C: Explicit job list
   jobs:
     - algorithm: "ppo"
       traffic: 100
       k_paths: 3
       network: "NSFNet"
     - algorithm: "dqn"
       traffic: 200
       k_paths: 5
       network: "COST239"

Required Grid Parameters
------------------------

These parameters MUST be present in grid specifications:

- ``path_algorithm``: Routing/RL algorithm name
- ``erlang_start``: Traffic load start value
- ``k_paths``: Number of candidate paths
- ``obs_space``: Observation space (for RL algorithms)
- ``network``: Network topology name (determines output grouping)

Fetch Configuration
-------------------

The ``configs/config.yml`` file for fetching results:

.. code-block:: yaml

   # Remote paths (user@host:path format)
   metadata_root: "user@cluster:/path/to/experiments"
   data_root: "user@cluster:/path/to/data"
   logs_root: "user@cluster:/path/to/logs"

   # Local destination
   dest: "~/cluster_results"

   # Experiment to fetch (relative path)
   experiment: "0119/1430/NSFNet"

   # Preview mode (true = don't actually download)
   dry_run: false

Output Format Reference
=======================

Manifest CSV
------------

One row per job, with all parameters as columns:

.. code-block:: text

   path_algorithm,erlang_start,erlang_stop,k_paths,network,num_requests,is_rl,...
   ppo,100,150,3,NSFNet,10000,true,...
   dqn,200,250,5,NSFNet,10000,true,...

**Encoding Rules:**

- Booleans: ``true`` / ``false`` (lowercase strings)
- Lists: JSON format ``[1,2,3]`` (no spaces)
- Dicts: JSON format ``{"key":"value"}`` (no spaces)
- Floats: No trailing zeros (``3.14`` not ``3.140000``)

Manifest Metadata JSON
----------------------

.. code-block:: json

   {
     "generated": "2025-01-19T14:30:45",
     "source": "/path/to/specs/my_experiment.yaml",
     "network": "NSFNet",
     "num_rows": 12,
     "resources": {
       "partition": "gpu-long",
       "time": "24:00:00",
       "mem": "32G",
       "cpus": 8,
       "gpus": 1,
       "nodes": 1
     }
   }

Directory Structure
-------------------

**On the cluster:**

.. code-block:: text

   FUSION/
     experiments/
       MMDD/
         HHMM/
           network/
             manifest.csv
             manifest_meta.json
             jobs/
               slurm_12345678_0.out
               slurm_12345678_1.out
     data/
       output/
         network/
           experiment_path/
             s1/
             s2/
       input/
         network/
           experiment_path/
             sim_input_s1.json
     logs/
       algorithm/
         network/
           MMDD/HHMM/

**Fetched locally:**

.. code-block:: text

   ~/cluster_results/
     data/
       network/
         experiment_path/
           output/
           input/
     logs/
       algorithm/
         network/

Components
==========

make_manifest.py
----------------

:Purpose: Generate job manifests from specification files
:Entry Point: ``python -m fusion.unity.make_manifest <spec>``

**Key Functions:**

- ``make_manifest(spec_path)``: Main entry point
- ``_expand_grid(grid, resources)``: Expand grid to job list
- ``_write_csv(rows, output_dir)``: Write manifest CSV
- ``_cast(key, value)``: Type cast parameters

submit_manifest.py
------------------

:Purpose: Submit manifest as SLURM array job
:Entry Point: ``python -m fusion.unity.submit_manifest <dir> <script>``

**Key Functions:**

- ``submit_manifest(experiment_dir, bash_script)``: Main entry point
- ``build_environment_variables()``: Create SLURM env vars
- ``read_first_row(manifest_path)``: Parse manifest header

fetch_results.py
----------------

:Purpose: Download results from cluster via rsync
:Entry Point: ``python -m fusion.unity.fetch_results``

**Key Functions:**

- ``fetch_results()``: Main entry point
- ``synchronize_remote_directory()``: rsync a directory
- ``convert_output_to_input_path()``: Path conversion

Helper Scripts
--------------

Located in ``fusion/unity/scripts/``:

- ``make_unity_venv.sh``: Create virtual environment on cluster
- ``priority.sh``: Check SLURM job priorities and queue
- ``group_jobs.sh``: Analyze resource usage by group

Error Handling
==============

The module defines a custom exception hierarchy:

.. code-block:: text

   UnityError (base)
   +-- ManifestError
   |   +-- ManifestNotFoundError
   |   +-- ManifestValidationError
   +-- SpecificationError
   |   +-- SpecNotFoundError
   |   +-- SpecValidationError
   +-- JobSubmissionError
   +-- SynchronizationError
   |   +-- RemotePathError
   +-- ConfigurationError

Common errors and solutions:

**SpecNotFoundError**: Specification file not found

- Check the file exists in current directory or ``specs/`` subdirectory
- Supported extensions: ``.yaml``, ``.yml``, ``.json``

**ManifestValidationError**: Invalid manifest parameters

- Ensure required fields are present (path_algorithm, erlang_start, etc.)
- Check parameter names match FUSION config schema

**JobSubmissionError**: SLURM submission failed

- Verify you're on the cluster with SLURM access
- Check partition name is valid for your cluster
- Ensure bash script exists in ``bash_scripts/`` directory

**SynchronizationError**: rsync failed

- Verify SSH access to cluster works
- Check paths in config.yml are correct
- Try with ``dry_run: true`` first

Testing
=======

:Test Location: ``fusion/unity/tests/``
:Run Tests: ``pytest fusion/unity/tests/ -v``

.. code-block:: bash

   # Run all unity tests
   pytest fusion/unity/tests/ -v

   # Run specific test file
   pytest fusion/unity/tests/test_make_manifest.py -v

Related Documentation
=====================

- :ref:`cli-module` - CLI entry points that may invoke unity commands
- :ref:`configs-module` - Configuration schema used for parameter validation
