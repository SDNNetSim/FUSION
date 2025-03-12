#!/bin/bash
#SBATCH -p cpu
#SBATCH -c 1
#SBATCH -G 0
#SBATCH --mem=16000
#SBATCH -t 0-12:00:00
#SBATCH --array=0-74
#SBATCH -o /dev/null  # Suppress SLURM’s default output

# -----------------------------------------------------------------------------
# Each traffic volume is run for 5 "variants":
#   0: epsilon_greedy_bandit
#   1: ucb_bandit
#   2: q_learning
#   3: ppo (default penalty from config, e.g. -10)
#   4: ppo (overridden penalty = -1)
# Traffic volumes: 50, 100, ..., 750 (15 total).
# Total tasks = 15 x 5 = 75 => --array=0-74
# -----------------------------------------------------------------------------

traffic_volumes=(50 100 150 200 250 300 350 400 450 500 550 600 650 700 750)
num_variants=5  # 0..4
total_tasks=${#traffic_volumes[@]}  # 15

# Determine traffic index and variant from the SLURM array ID.
traffic_idx=$(( SLURM_ARRAY_TASK_ID / num_variants ))
variant=$(( SLURM_ARRAY_TASK_ID % num_variants ))
erlang_start=${traffic_volumes[$traffic_idx]}

# Assign algorithm/hyperparams based on variant.
case $variant in
    0)
        # epsilon_greedy_bandit with recommended hyperparams
        alg="epsilon_greedy_bandit"
        extra_params="--penalty -10 --epsilon_start 0.30 --epsilon_end 0.05 --decay_rate 0.22"
        ;;
    1)
        # ucb_bandit with recommended hyperparams
        alg="ucb_bandit"
        extra_params="--penalty -10 --conf_param 3.0"
        ;;
    2)
        # q_learning with recommended hyperparams
        alg="q_learning"
        extra_params="--penalty -10 --alpha_start 0.15 --alpha_end 0.08 --epsilon_start 0.09 --epsilon_end 0.07 --gamma 0.90 --decay_rate 0.28"
        ;;
    3)
        # ppo, default penalty from config
        alg="ppo"
        extra_params=""
        ;;
    4)
        # ppo, override penalty to -1
        alg="ppo"
        extra_params="--penalty -1"
        ;;
    *)
        echo "Error: unknown variant $variant"
        exit 1
        ;;
esac

# Compute erlang_stop (same logic as before).
erlang_stop=$(( erlang_start + 100 ))

# -----------------------------------------------------------------------------
# Set up structured logging.
# -----------------------------------------------------------------------------
BASE_LOG_DIR="slurm_logs"
output_dir="${BASE_LOG_DIR}/${alg}/${erlang_start}"
mkdir -p "$output_dir"

timestamp=$(date +"%Y-%m-%d_%H-%M-%S")
output_file="${output_dir}/slurm_output_${timestamp}.out"

exec > "$output_file" 2>&1

# -----------------------------------------------------------------------------
# Start job
# -----------------------------------------------------------------------------
echo "Job started at $(date)"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "SLURM Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Algorithm: $alg"
echo "Erlang start: $erlang_start"
echo "Variant: $variant"
echo "----------------------------------------------------------"

DEFAULT_DIR="/work/pi_vinod_vokkarane_uml_edu/git/FUSION"
cd "$DEFAULT_DIR"
echo "Working directory: $(pwd)"

module load python/3.11.7

VENV_DIR="venvs/unity_venv"
if [ ! -d "$VENV_DIR" ]; then
  echo "Directory '$VENV_DIR' does not exist. Creating it now..."
  mkdir -p "$VENV_DIR"
  ./bash_scripts/make_unity_venv.sh "$VENV_DIR" python3.11
fi

if [ ! -d "$VENV_DIR/venv" ]; then
  echo "Virtual environment creation failed. Exiting."
  exit 1
fi

source "$VENV_DIR/venv/bin/activate"
pip install -r requirements.txt

# Register your custom RL environment(s).
./bash_scripts/register_rl_env.sh ppo SimEnv

# -----------------------------------------------------------------------------
# Run the simulation
# -----------------------------------------------------------------------------
echo "Running simulation with Erlang range: $erlang_start to $erlang_stop"
echo "Extra parameters: $extra_params"
echo "----------------------------------------------------------"

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
