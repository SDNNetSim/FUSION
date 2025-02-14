#!/bin/bash

#SBATCH -p cpu
#SBATCH -c 1
#SBATCH -G 0
#SBATCH --mem=32000
#SBATCH -t 1-00:00:00
#SBATCH -o slurm-%A_%a.out  # Output file for each task
#SBATCH --array=0

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

python -m rl_zoo3.train --algo ppo --env SimEnv --conf-file ./sb3_scripts/yml/ppo.yml -optimize --n-trials 50 --n-timesteps 250000
