#!/bin/bash

#SBATCH -p arm-preempt
#SBATCH -c 20
#SBATCH -G 0
#SBATCH --mem=40000
#SBATCH -t 14-00:00:00
#SBATCH -o slurm-%A_%a.out  # Output file for each task
#SBATCH --array=0-3  # Define array size based on total combinations

# This script is designed to run a reinforcement learning simulation on the Unity cluster at UMass Amherst.
# It sets up the necessary environment, installs dependencies, registers custom environments, and runs
# the simulation with different parameter combinations. Note that it uses the SLURM job scheduler.

# Ensure the script stops if any command fails
set -e

# Change to the home directory and then to the SDN simulator directory
# shellcheck disable=SC2164
cd
# shellcheck disable=SC2164
# Default directory
DEFAULT_DIR="/work/pi_vinod_vokkarane_uml_edu/git/sdn_simulator/"

# Check for user input
if [ -z "$1" ]; then
  echo "No directory provided. Using default directory: $DEFAULT_DIR"
  cd "$DEFAULT_DIR"
else
  echo "Changing to user-specified directory: $1"
  cd "$1"
fi

echo "Current directory: $(pwd)"

module load python/3.11.7

# Activate the virtual environment
# Create the virtual environment if it doesn't exist
if [ ! -d "venvs/unity_venv/venv" ]; then
  ./bash_scripts/make_venv.sh venvs/unity_venv python3.11
fi
source venvs/unity_venv/venv/bin/activate

# Install the required Python packages
pip install -r requirements.txt

# Register custom reinforcement learning environments in Stable Baselines 3
./bash_scripts/register_rl_env.sh ppo SimEnv

# Declare arrays for parameter values
algorithm_list=("ucb_bandit" "epsilon_greedy_bandit")
network_list=("NSFNet")
k_paths_list=("3")
epsilon_start_list=("0.01" "0.06" "0.1" "0.2")
reward_list=("1" "10" "100")
penalty_list=("-1" "-10" "-100")

# Calculate the number of combinations
num_algorithms=${#algorithm_list[@]}
num_networks=${#network_list[@]}
num_k_paths=${#k_paths_list[@]}
num_epsilons=${#epsilon_start_list[@]}
num_rewards=${#reward_list[@]}
num_penalties=${#penalty_list[@]}
total_combinations=$((num_algorithms * num_networks * num_k_paths * num_epsilons * num_rewards * num_penalties))

# Calculate the indices for each parameter
alg_idx=$((SLURM_ARRAY_TASK_ID % num_algorithms))
network_idx=$(((SLURM_ARRAY_TASK_ID / num_algorithms) % num_networks))
k_idx=$(((SLURM_ARRAY_TASK_ID / (num_algorithms * num_networks)) % num_k_paths))
eps_idx=$(((SLURM_ARRAY_TASK_ID / (num_algorithms * num_networks * num_k_paths)) % num_epsilons))
reward_idx=$(((SLURM_ARRAY_TASK_ID / (num_algorithms * num_networks * num_k_paths * num_epsilons)) % num_rewards))
penalty_idx=$(((SLURM_ARRAY_TASK_ID / (num_algorithms * num_networks * num_k_paths * num_epsilons * num_rewards)) % num_penalties))

# Extract parameters based on the computed indices
alg="${algorithm_list[$alg_idx]}"
network="${network_list[$network_idx]}"
k="${k_paths_list[$k_idx]}"
eps="${epsilon_start_list[$eps_idx]}"
reward="${reward_list[$reward_idx]}"
penalty="${penalty_list[$penalty_idx]}"

# Run the Python script with the extracted parameters
python run_rl_sim.py \
  --network "$network" \
  --k_paths "$k" \
  --epsilon_start "$eps" \
  --reward "$reward" \
  --penalty "$penalty" \
  --path_algorithm "$alg" \
  --core_algorithm first_fit ||
  echo "Error running Python script for job $SLURM_ARRAY_TASK_ID"
