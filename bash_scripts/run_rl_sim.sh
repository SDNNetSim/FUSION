#!/bin/bash

#SBATCH -p cpu
#SBATCH -c 1
#SBATCH -G 0
#SBATCH --mem=16000
#SBATCH -t 0-5:00:00
#SBATCH --array=0-8   # 3 traffic volumes * 3 algorithms
#SBATCH -o /dev/null  # Disable default SLURM output redirection

# Stop the script if any command fails
set -e

# Change to the default directory or a user-specified directory
cd
DEFAULT_DIR="/work/pi_vinod_vokkarane_uml_edu/git/FUSION/"
if [ -z "$1" ]; then
  echo "No directory provided. Using default directory: $DEFAULT_DIR"
  cd "$DEFAULT_DIR"
else
  echo "Changing to user-specified directory: $1"
  cd "$1"
fi

echo "Current directory: $(pwd)"

# Load the required Python module
module load python/3.11.7

# Activate (or create if necessary) the virtual environment
if [ ! -d "venvs/unity_venv/venv" ]; then
  ./bash_scripts/make_unity_venv.sh venvs/unity_venv python3.11
  ./bash_scripts/make_unity_venv.sh venvs/unity_venv python3.11
fi
source venvs/unity_venv/venv/bin/activate

# Install required Python packages
pip install -r requirements.txt

# (Re)register the custom RL environments
./bash_scripts/register_rl_env.sh ppo SimEnv

# Define the algorithms
algorithms=("epsilon_greedy_bandit" "ucb_bandit" "q_learning")
traffic_volumes=(50 400 750)  # Only the three specified traffic volumes
num_algs=${#algorithms[@]}
num_traffic=${#traffic_volumes[@]}

# Decode SLURM_ARRAY_TASK_ID
alg_idx=$(( SLURM_ARRAY_TASK_ID % num_algs ))
traffic_idx=$(( SLURM_ARRAY_TASK_ID / num_algs ))

# Assign algorithm and traffic volume
alg=${algorithms[$alg_idx]}
erlang_start=${traffic_volumes[$traffic_idx]}
erlang_stop=$((erlang_start + 100))  # Keeping erlang_step 100

echo "Running simulation with:"
echo "  Algorithm:      $alg"
echo "  Erlang range:   $erlang_start to $erlang_stop"

# No hyperparameters are defined; extra_params is empty
extra_params=""

echo "Extra parameters: $extra_params"

# Generate timestamp for unique output file
timestamp=$(date +"%Y-%m-%d_%H-%M-%S.%N" | cut -c1-26)

# Create a directory structure based on the algorithm and erlang_start value
output_dir="bash_scripts/slurm/${alg}/${erlang_start}"
mkdir -p "$output_dir"
output_file="${output_dir}/slurm_${SLURM_JOB_ID}_${alg}_${timestamp}.out"

# Run the simulation and redirect stdout and stderr to the output file
{
  echo "Job ID: $SLURM_JOB_ID"
  echo "Algorithm: $alg"
  echo "Erlang start: $erlang_start"
  echo "Erlang stop: $erlang_stop"
  echo "Extra parameters: $extra_params"
  echo "Timestamp: $timestamp"
  echo "-------------------------"

  python run_rl_sim.py \
    --erlang_start "$erlang_start" \
    --erlang_stop "$erlang_stop" \
    --erlang_step 100 \
    --path_algorithm "$alg" || echo "Error running Python script for algorithm $alg"

  echo "-------------------------"
  echo "Finished simulation for $alg"
} &> "$output_file"

