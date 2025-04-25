#!/bin/bash
echo "Launching job ${SLURM_ARRAY_TASK_ID}"
echo "Using manifest: ${MANIFEST}"
echo "Output dir: ${JOB_DIR}"

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
  source "$VENV_DIR/venv/bin/activate"
fi

mapfile -t LINES < "$MANIFEST"
HEADER="${LINES[0]}"
ROW="${LINES[$((SLURM_ARRAY_TASK_ID+1))]}"

echo "CSV row ${SLURM_ARRAY_TASK_ID}: $ROW"
echo "Raw header: $HEADER"

IFS=',' read -ra COL_NAMES <<<"$HEADER"
IFS=',' read -ra COL_VALUES <<<"$ROW"

if [[ "${#COL_NAMES[@]}" -ne "${#COL_VALUES[@]}" ]]; then
  echo "ERROR: Column mismatch!"
  echo "Header columns: ${#COL_NAMES[@]}"
  echo "Row columns: ${#COL_VALUES[@]}"
  exit 1
fi

declare -A EXCLUDE_MAP=(
  ["run_id"]=1
  ["partition"]=1
  ["cpus"]=1
  ["mem"]=1
  ["time"]=1
  ["gpus"]=1
  ["nodes"]=1
  ["is_rl"]=1
)

ARGS=""
for i in "${!COL_NAMES[@]}"; do
  raw_key="${COL_NAMES[$i]}"
  raw_val="${COL_VALUES[$i]}"
  key="$(echo "$raw_key" | tr -d '\r\n' | xargs)"
  val="$(echo "$raw_val" | tr -d '\r\n' | xargs)"

  echo "Checking: key='$key', val='$val'"

  if [[ -z "${EXCLUDE_MAP[$key]}" && -n "$key" ]]; then
    ARGS+="--${key} ${val} "
  else
    echo "Skipping key: $key"
  fi
done

ARGS="--run_id ${SLURM_ARRAY_TASK_ID} ${ARGS}"

PY_OUT=$(python run_rl_sim.py ${ARGS})
echo "$PY_OUT"
RESULT_PATH=$(echo "$PY_OUT" | awk -F= '/^OUTPUT_DIR=/{print $2}')
echo "Mapped run_id=${SLURM_ARRAY_TASK_ID} → ${RESULT_PATH}"
echo "{\"run_id\":\"${SLURM_ARRAY_TASK_ID}\",\"path\":\"${RESULT_PATH}\"}" \
     >> "unity/${JOB_DIR}/runs_index.json"

