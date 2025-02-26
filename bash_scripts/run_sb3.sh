#!/bin/bash

#SBATCH -p cpu
#SBATCH -c 1
#SBATCH -G 0
#SBATCH --mem=32000
#SBATCH -t 0-10:00:00
#SBATCH --array=0-2   # One job per traffic volume: 50, 400, 750
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

# Define the algorithm (in this case, ppo)
alg="ppo"

# Define the traffic volumes
traffic_volumes=(50 400 750)

# Use SLURM_ARRAY_TASK_ID to select the traffic volume
erlang_start=${traffic_volumes[$SLURM_ARRAY_TASK_ID]}
erlang_stop=$((erlang_start + 100))  # Keeping erlang_step 100

echo "Running simulation with:"
echo "  Algorithm:      $alg"
echo "  Erlang range:   $erlang_start to $erlang_stop"

# Generate a timestamp with picosecond precision
timestamp=$(date +"%Y-%m-%d_%H-%M-%S.%N" | cut -c1-26)

# Create an output directory based on the algorithm and traffic volume
output_dir="bash_scripts/slurm/${alg}/${erlang_start}"
mkdir -p "$output_dir"

# Define the output file with timestamp
output_file="${output_dir}/slurm_${SLURM_JOB_ID}_${alg}_${timestamp}.out"

# Run the simulation and redirect stdout and stderr to the output file
{
  echo "Job ID: $SLURM_JOB_ID"
  echo "Algorithm: $alg"
  echo "Erlang start: $erlang_start"
  echo "Erlang stop: $erlang_stop"
  echo "-------------------------"
  
  python run_rl_sim.py \
    --erlang_start "$erlang_start" \
    --erlang_stop "$erlang_stop" \
    --erlang_step 100 \
    --path_algorithm "$alg"
  
  echo "-------------------------"
  echo "Finished simulation for $alg with traffic volume $erlang_start"
} &> "$output_file"
