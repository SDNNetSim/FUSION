#!/bin/bash

#SBATCH -p cpu
#SBATCH -c 1
#SBATCH -G 0
#SBATCH --mem=32000
#SBATCH -t 0-5:00:00
#SBATCH --array=0-14   # 15 traffic volumes * 4 algorithms = 60 jobs
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

# Define the algorithms and traffic volumes
algorithms=("ppo")
traffic_volumes=($(seq 50 50 750))   # Generates: 50 100 150 ... 750
num_algs=${#algorithms[@]}
num_traffic=${#traffic_volumes[@]}

# Decode SLURM_ARRAY_TASK_ID into algorithm index and traffic volume index
alg_idx=$(( SLURM_ARRAY_TASK_ID % num_algs ))
traffic_idx=$(( SLURM_ARRAY_TASK_ID / num_algs ))

# Assign algorithm and traffic volume
alg=${algorithms[$alg_idx]}
erlang_start=${traffic_volumes[$traffic_idx]}
erlang_stop=$(( erlang_start + 100 ))  # Each job gets a 100-unit span

echo "Running simulation with:"
echo "  Algorithm:      $alg"
echo "  Erlang range:   $erlang_start to $erlang_stop"

# Set extra parameters based on the algorithm
if [ "$alg" = "epsilon_greedy_bandit" ]; then
  extra_params="--epsilon_start 0.27 --epsilon_end 0.04 --decay_rate 0.29"
elif [ "$alg" = "ucb_bandit" ]; then
  extra_params="--conf_param 2.2"
elif [ "$alg" = "q_learning" ]; then
  extra_params="--alpha_start 0.18 --alpha_end 0.06 --epsilon_start 0.32 --epsilon_end 0.06 --gamma 0.93 --decay_rate 0.30"
elif [ "$alg" = "ppo" ]; then
  extra_params=""   # For PPO, you'll update hyperparameters via YAML later
fi

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
    --path_algorithm "$alg" \
    $extra_params || echo "Error running Python script for algorithm $alg"

  echo "-------------------------"
  echo "Finished simulation for $alg with Erlang range $erlang_start to $erlang_stop"
} &> "$output_file"
