#!/bin/bash
#SBATCH -p cpu
#SBATCH -c 1
#SBATCH -G 0
#SBATCH --mem=32000
#SBATCH -t 2-00:00:00   # Job time increased to 3 days
#SBATCH -o /dev/null   # Suppress SLURM’s default output

# -----------------------------------------------------------------------------
# Compute log variables as early as possible (before any output occurs)
# -----------------------------------------------------------------------------
BASE_LOG_DIR="slurm_logs"
alg="ppo"
output_dir="${BASE_LOG_DIR}/${alg}/optimize"
mkdir -p "$output_dir"

timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
output_file="${output_dir}/slurm_output_${timestamp}.out"

# Redirect all subsequent output (stdout & stderr) to our computed log file.
exec > "$output_file" 2>&1

echo "Job started at $(date)"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Algorithm: $alg"
echo "Running in optimization mode"
echo "----------------------------------------------------------"

# Set the working directory to your repository root.
DEFAULT_DIR="/work/pi_vinod_vokkarane_uml_edu/git/FUSION"
cd "$DEFAULT_DIR"
echo "Working directory: $(pwd)"

# Load the required Python module
module load python/3.11.7

# Ensure the virtual environment directory exists in the repository root.
VENV_DIR="venvs/unity_venv"
if [ ! -d "$VENV_DIR" ]; then
  echo "Directory '$VENV_DIR' does not exist. Creating it now..."
  mkdir -p "$VENV_DIR"
  ./bash_scripts/make_unity_venv.sh "$VENV_DIR" python3.11
fi

# Check that the virtual environment was created successfully.
if [ ! -d "$VENV_DIR/venv" ]; then
  echo "Virtual environment creation failed. Exiting."
  exit 1
fi

# Activate the virtual environment.
source "$VENV_DIR/venv/bin/activate"

# Install required Python packages.
pip install -r requirements.txt

# (Re)register the custom RL environments.
./bash_scripts/register_rl_env.sh ppo SimEnv

# Run the optimization command.
echo "Running optimization command:"
python -m rl_zoo3.train --algo ppo --env SimEnv --conf-file ./sb3_scripts/yml/ppo.yml \
  -optimize --n-trials 20 --n-timesteps 250000 || echo "Error running optimization for algorithm $alg"

echo "----------------------------------------------------------"
echo "Finished optimization for $alg"
echo "Job ended at $(date)"
echo "Output saved to $output_file"
