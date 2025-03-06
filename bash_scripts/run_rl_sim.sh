#!/bin/bash
#SBATCH -p cpu
#SBATCH -c 1
#SBATCH -G 0
#SBATCH --mem=16000
#SBATCH -t 0-0:25:00
#SBATCH --array=0
#SBATCH -o /dev/null  # Suppress SLURM’s default output

# -----------------------------------------------------------------------------
# Compute log variables as early as possible (before any output occurs)
# -----------------------------------------------------------------------------
# Define the algorithms and traffic volumes (adjust as needed)
algorithms=("epsilon_greedy_bandit")
traffic_volumes=($(seq 50 50 750))
num_algs=${#algorithms[@]}

# Determine which algorithm and traffic volume this task uses.
alg_idx=$(( SLURM_ARRAY_TASK_ID % num_algs ))
traffic_idx=$(( SLURM_ARRAY_TASK_ID / num_algs ))
alg=${algorithms[$alg_idx]}
erlang_start=${traffic_volumes[$traffic_idx]}

# Set up the structured log directory: slurm_logs/<algorithm>/<erlang_start>/
BASE_LOG_DIR="slurm_logs"
output_dir="${BASE_LOG_DIR}/${alg}/${erlang_start}"
mkdir -p "$output_dir"

# Create a timestamped log file name.
timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
output_file="${output_dir}/slurm_output_${timestamp}.out"

# Redirect all subsequent output (stdout & stderr) to our computed log file.
exec > "$output_file" 2>&1

# -----------------------------------------------------------------------------
# Now continue with the rest of your job. All output now goes to $output_file.
# -----------------------------------------------------------------------------

echo "Job started at $(date)"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "SLURM Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Algorithm: $alg"
echo "Erlang start: $erlang_start"
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

# Define additional simulation parameters.
erlang_stop=$(( erlang_start + 100 ))
echo "Running simulation with Erlang range: $erlang_start to $erlang_stop"

if [ "$alg" = "epsilon_greedy_bandit" ]; then
  extra_params="--epsilon_start 0.27 --epsilon_end 0.04 --decay_rate 0.29"
elif [ "$alg" = "ucb_bandit" ]; then
  extra_params="--conf_param 2.2"
elif [ "$alg" = "q_learning" ]; then
  extra_params="--alpha_start 0.18 --alpha_end 0.06 --epsilon_start 0.32 --epsilon_end 0.06 --gamma 0.93 --decay_rate 0.30"
else
  extra_params=""
fi
echo "Extra parameters: $extra_params"
echo "----------------------------------------------------------"

# Run the simulation.
python run_rl_sim.py \
  --erlang_start "$erlang_start" \
  --erlang_stop "$erlang_stop" \
  --erlang_step 100 \
  --path_algorithm "$alg" \
  $extra_params || echo "Error running Python script for algorithm $alg"

echo "----------------------------------------------------------"
echo "Finished simulation for $alg with Erlang range $erlang_start to $erlang_stop"
echo "Job ended at $(date)"
echo "Output saved to $output_file"
