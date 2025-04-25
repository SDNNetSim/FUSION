#!/bin/bash
#SBATCH -p cpu
#SBATCH -c 1
#SBATCH --mem=32000
#SBATCH -t 0-00:30:00
#SBATCH --array=0-__N_JOBS__
#SBATCH -o __JOB_DIR__/slurm_%A_%a.out

module load python/3.11.7


DEFAULT_DIR="/work/pi_vinod_vokkarane_uml_edu/git/FUSION"
PROJECT_DIR="${1:-$DEFAULT_DIR}"

cd
cd "$PROJECT_DIR" || { echo "Directory not found: $PROJECT_DIR"; exit 1; }
echo "Current working directory: $(pwd)"

VENV_DIR="venvs/unity_venv"
SCRIPTS_DIR="unity/bash_scripts"
RL_ALGS=(ppo a2c dqn qr_dqn)
ENV_IDS=(SimEnv SimEnv SimEnv SimEnv)

if [[ ! -d "$VENV_DIR" ]]; then
  echo "Creating Unity venv..."
  mkdir -p "$VENV_DIR"
  bash "$SCRIPTS_DIR/make_unity_venv.sh" "$VENV_DIR" python3.11
  echo "Activating venv and installing requirements..."
  source "$VENV_DIR/venv/bin/activate"
  pip install -r requirements.txt
  echo "Registering RL environments..."
  for i in "${!RL_ALGS[@]}"; do
    bash "$SCRIPTS_DIR/register_rl_env.sh" "${RL_ALGS[$i]}" "${ENV_IDS[$i]}"
  done
else
  echo "Venv already exists; activating..."
  source "$VENV_DIR/bin/activate"
fi

# Pull row #SLURM_ARRAY_TASK_ID+2 because CSV has a header line ---------------
ROW=$(sed -n "$((SLURM_ARRAY_TASK_ID+2))p" "$MANIFEST")
IFS=',' read -r run_id algorithm t_start t_stop k_paths seed is_rl <<<"$ROW"

# Where your classic results go (unchanged) ----------------------------------
now_ts=$(date +%H%M%S_%N)                         # keeps the picoseconds if you like
RES_DIR="data/output/${NETWORK}/${DATE}/${now_ts}/${t_start}"
mkdir -p "$RES_DIR"

python run_rl_sim.py \
    --erlang_start "$t_start"  --erlang_stop "$t_stop" \
    --network "$NETWORK"       --k_paths "$k_paths" \
    --path_algorithm "$algorithm" \
    --seed "$seed"             --run_id "$run_id" \
    --output_dir "$RES_DIR" \
 && touch "${RES_DIR}/DONE"
