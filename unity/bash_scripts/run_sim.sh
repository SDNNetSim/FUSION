#!/bin/bash

#SBATCH -p arm-preempt
#SBATCH -c 20
#SBATCH -G 0
#SBATCH --mem=40000
#SBATCH -t 14-00:00:00
#SBATCH -o slurm-%A_%a.out  # Output file for each task
#SBATCH --array=0-3  # Define array size based on total combinations

# This script is designed to run a non-artificial intelligence simulation on the Unity cluster at UMass Amherst.

set -e # Stop the script if any command fails

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

# Confirm current directory
echo "Current directory: $(pwd)"

echo "Starting job with SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"

# Load the Python 3.11.7 module
module load python/3.11.7

# Verify the loaded Python version
python --version

# Activate the virtual environment
# Create the virtual environment if it doesn't exist
if [ ! -d "venvs/unity_venv/venv" ]; then
  ./bash_scripts/make_venv.sh venvs/unity_venv python3.11
fi
source venvs/unity_venv/venv/bin/activate

# Install the required Python packages
pip install -r requirements.txt

# Define parameter arrays
allocation_methods=("first_fit" "last_fit")
spectrum_allocation_priorities=("CSB" "BSC")
cores_per_link=(4)

# Define additional variables
num_requests=15000
multi_fiber=False

# Calculate total combinations
total_combinations=$((${#allocation_methods[@]} * ${#spectrum_allocation_priorities[@]} * ${#cores_per_link[@]}))

# Ensure the task ID is within the range of total combinations
if [ "$SLURM_ARRAY_TASK_ID" -ge "$total_combinations" ]; then
  echo "SLURM_ARRAY_TASK_ID out of range."
  exit 1
fi

# Calculate indices based on SLURM_ARRAY_TASK_ID
allocation_method_index=$((SLURM_ARRAY_TASK_ID / (${#spectrum_allocation_priorities[@]} * ${#cores_per_link[@]})))
temp_index=$((SLURM_ARRAY_TASK_ID % (${#spectrum_allocation_priorities[@]} * ${#cores_per_link[@]})))
spectrum_priority_index=$((temp_index / ${#cores_per_link[@]}))
cores_per_link_index=$((temp_index % ${#cores_per_link[@]}))

# Select parameters for this task
allocation_method=${allocation_methods[$allocation_method_index]}
spectrum_priority=${spectrum_allocation_priorities[$spectrum_priority_index]}
cores=${cores_per_link[$cores_per_link_index]}

# Print parameters for this task
echo "Running simulation with:"
echo "  Allocation method: $allocation_method"
echo "  Spectrum priority: $spectrum_priority"
echo "  Cores per link: $cores"
echo "  Number of requests: $num_requests"
echo "  Multi-fiber: $multi_fiber"

# Run the simulation with the specified parameters
python run_sim.py --allocation_method "$allocation_method" \
  --spectrum_allocation_priority "$spectrum_priority" \
  --cores_per_link "$cores" \
  --num_requests "$num_requests" \
  --multi_fiber "$multi_fiber"

echo "Job completed successfully for SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
