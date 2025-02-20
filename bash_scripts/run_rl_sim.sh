#!/bin/bash

#SBATCH -p cpu
#SBATCH -c 1
#SBATCH -G 0
#SBATCH --mem=32000
#SBATCH -t 1-0:00:00
#SBATCH --array=0-7
#SBATCH -o /dev/null   # Disable default SLURM output redirection

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

# Define the algorithms and determine which one to run for this job
algorithms=("ppo")
num_algs=${#algorithms[@]}

# Decode the SLURM_ARRAY_TASK_ID:
#   - algorithm index is (array index modulo number of algorithms)
#   - erlang index is (array index divided by number of algorithms)
alg_idx=$(( SLURM_ARRAY_TASK_ID % num_algs ))
erlang_idx=$(( SLURM_ARRAY_TASK_ID / num_algs ))

alg=${algorithms[$alg_idx]}

# Calculate the erlang range for this job:
#   Erlang start = 50 + (erlang index * 100)
#   Erlang stop  = erlang_start + 100  (so 750 yields 750 to 850)
erlang_start=$((50 + erlang_idx * 100))
erlang_stop=$((erlang_start + 100))

echo "Running simulation with:"
echo "  Algorithm:      $alg"
echo "  Erlang range:   $erlang_start to $erlang_stop"

# Set extra parameters based on the algorithm
if [ "$alg" == "epsilon_greedy_bandit" ]; then
  extra_params="--epsilon_start 0.4 --epsilon_end 0.07 --decay_rate 0.47 --epsilon_update exp_decay"
elif [ "$alg" == "ucb_bandit" ]; then
  extra_params="--conf_param 4.8"
elif [ "$alg" == "q_learning" ]; then
  extra_params="--alpha_start 0.3 --alpha_end 0.06 --epsilon_start 0.35 --epsilon_end 0.06 --gamma 0.91 --decay_rate 0.3 --epsilon_update exp_decay --alpha_update linear_decay"
fi

echo "Extra parameters: $extra_params"

# Create a directory structure based on the algorithm and erlang_start value
output_dir="bash_scripts/slurm/${alg}/${erlang_start}"
mkdir -p "$output_dir"
output_file="${output_dir}/slurm_${SLURM_JOB_ID}_${alg}.out"

# Run the simulation and redirect stdout and stderr to the output file
{
  echo "Job ID: $SLURM_JOB_ID"
  echo "Algorithm: $alg"
  echo "Erlang start: $erlang_start"
  echo "Erlang stop: $erlang_stop"
  echo "Extra parameters: $extra_params"
  echo "-------------------------"
  
  python run_rl_sim.py \
    --erlang_start "$erlang_start" \
    --erlang_stop "$erlang_stop" \
    --erlang_step 100 \
    --path_algorithm "$alg" \
    $extra_params || echo "Error running Python script for algorithm $alg"
  
  echo "-------------------------"
  echo "Finished simulation for $alg"
} &> "$output_file"
