#!/bin/bash

#SBATCH -p cpu
#SBATCH -c 1
#SBATCH -G 0
#SBATCH --mem=32000
#SBATCH -t 0-10:00:00   # Job time increased to 10 hours
#SBATCH --array=0-2     # Only used in simulation mode
# (No -o flag so that SLURM's default output file is used)

# Exit immediately if a command fails
set -e

# Set the default directory
DEFAULT_DIR="/work/pi_vinod_vokkarane_uml_edu/git/FUSION/"

# Parse arguments:
# If the first argument exists and does not start with "--", use it as the working directory.
# Otherwise, use DEFAULT_DIR.
if [ "$#" -ge 1 ] && [[ "$1" != --* ]]; then
  echo "Changing to user-specified directory: $1"
  cd "$1"
else
  echo "No valid directory provided. Using default directory: $DEFAULT_DIR"
  cd "$DEFAULT_DIR"
fi

# Determine if optimize flag was provided. Look at second argument if first argument was used,
# or first argument if no directory was provided.
if [ "$#" -ge 2 ]; then
  optimize_flag="$2"
elif [ "$#" -eq 1 ] && [[ "$1" == --* ]]; then
  optimize_flag="$1"
else
  optimize_flag=""
fi

# Debug: Print SLURM variables and working directory at start
echo "===== Starting Job ====="
echo "SLURM_JOB_ID: ${SLURM_JOB_ID}"
echo "SLURM_ARRAY_TASK_ID: ${SLURM_ARRAY_TASK_ID}"
echo "Current working directory: $(pwd)"
echo "Optimize flag: ${optimize_flag}"
echo "========================"

# Load required Python module
module load python/3.11.7

# Activate (or create if necessary) the virtual environment
if [ ! -d "venvs/unity_venv/venv" ]; then
  ./bash_scripts/make_unity_venv.sh venvs/unity_venv python3.11
  ./bash_scripts/make_unity_venv.sh venvs/unity_venv python3.11
fi
source venvs/unity_venv/venv/bin/activate

# Install required packages
pip install -r requirements.txt

# (Re)register custom RL environments
./bash_scripts/register_rl_env.sh ppo SimEnv

# Define the algorithm (ppo)
alg="ppo"

# Determine mode: simulation or optimization based on the optimize_flag
if [ "$optimize_flag" = "--optimize" ]; then
  mode="optimize"
else
  mode="simulation"
fi

# Set a timestamp for our custom log file
timestamp=$(date +"%Y-%m-%d_%H-%M-%S.%N" | cut -c1-26)

if [ "$mode" = "simulation" ]; then
  # Define traffic volumes and pick one based on SLURM_ARRAY_TASK_ID
  traffic_volumes=(50 400 750)
  erlang_start=${traffic_volumes[$SLURM_ARRAY_TASK_ID]}
  erlang_stop=$((erlang_start + 100))  # Keep erlang_step as 100
  
  echo "Running simulation with:"
  echo "  Algorithm:      $alg"
  echo "  Erlang range:   $erlang_start to $erlang_stop"
  
  output_dir="bash_scripts/slurm/${alg}/${erlang_start}"
  output_file="${output_dir}/slurm_${SLURM_JOB_ID}_${alg}_${timestamp}.out"
else
  # For optimization mode, ignore SLURM_ARRAY_TASK_ID
  echo "Running optimization for algorithm: $alg"
  
  output_dir="bash_scripts/slurm/${alg}/optimize"
  output_file="${output_dir}/slurm_${SLURM_JOB_ID}_${alg}_optimize_${timestamp}.out"
fi

# Create the output directory for our custom log file
mkdir -p "$output_dir"
echo "Custom log file will be: $output_file"

# Write a header to the custom log file for debugging
{
  echo "===== Custom Log File for Job ${SLURM_JOB_ID} ====="
  echo "Mode: $mode"
  echo "Timestamp: $timestamp"
  if [ "$mode" = "simulation" ]; then
    echo "Erlang start: $erlang_start"
    echo "Erlang stop: $erlang_stop"
  fi
  echo "======================================"
} > "$output_file"

# Run the appropriate command and tee output to our custom log file
{
  echo "Job ID: $SLURM_JOB_ID"
  echo "Algorithm: $alg"
  if [ "$mode" = "simulation" ]; then
    echo "Erlang start: $erlang_start"
    echo "Erlang stop: $erlang_stop"
    echo "-------------------------"
    
    python run_rl_sim.py \
      --erlang_start "$erlang_start" \
      --erlang_stop "$erlang_stop" \
      --erlang_step 100 \
      --path_algorithm "$alg"
  else
    echo "Mode: Optimization"
    echo "-------------------------"
    
    python -m rl_zoo3.train --algo ppo --env SimEnv --conf-file ./sb3_scripts/yml/ppo.yml \
      -optimize --n-trials 15 --n-timesteps 250000
  fi
  
  echo "-------------------------"
  echo "Finished job for $alg in $mode mode"
} 2>&1 | tee -a "$output_file"
