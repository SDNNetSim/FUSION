#!/bin/bash

#SBATCH -p cpu
#SBATCH -c 1
#SBATCH -G 0
#SBATCH --mem=16000
#SBATCH -t 1-00:00:00
#SBATCH -o slurm-%A_%a.out  # Output file for each task
#SBATCH --array=0-8

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

algorithm_list=("epsilon_greedy_bandit" "ucb_bandit" "q_learning")
erlang_list=("50" "250" "700")

num_algorithms=${#algorithm_list[@]}
num_erlangs=${#erlang_list[@]}
total_combinations=$((num_algorithms * num_erlangs))

alg_idx=$(( SLURM_ARRAY_TASK_ID % num_algorithms ))
erlang_idx=$(( SLURM_ARRAY_TASK_ID / num_algorithms ))

alg="${algorithm_list[$alg_idx]}"
erlang_start="${erlang_list[$erlang_idx]}"
erlang_stop=$(( erlang_start + 50 ))

extra_params=""
if [ "$alg" == "epsilon_greedy_bandit" ]; then
  extra_params="--epsilon_update exp_decay"
elif [ "$alg" == "q_learning" ]; then
  extra_params="--epsilon_update exp_decay --alpha_update linear_decay"
fi

echo "Running simulation with:"
echo "  path_algorithm: $alg"
echo "  erlang_start:   $erlang_start"
echo "  erlang_stop:    $erlang_stop"
echo "  extra_params:   $extra_params"

# --- Execute the Python simulation script with the computed parameters ---
python run_rl_sim.py \
  --erlang_start "$erlang_start" \
  --erlang_stop "$erlang_stop" \
  --erlang_step 50 \
  --path_algorithm "$alg" \
  --core_algorithm first_fit \
  $extra_params || echo "Error running Python script for job $SLURM_ARRAY_TASK_ID"

