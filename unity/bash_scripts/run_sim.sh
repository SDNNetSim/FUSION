#!/bin/bash

echo
echo "==========================================="
echo "🧪 Starting SLURM Job ${SLURM_ARRAY_TASK_ID}"
echo "Manifest: ${MANIFEST}"
echo "Job Directory: ${JOB_DIR}"
echo "==========================================="
echo

module load python/3.11.7

# --- Project Setup ---
DEFAULT_DIR="/work/pi_vinod_vokkarane_uml_edu/git/sdn_simulator"
PROJECT_DIR="${1:-$DEFAULT_DIR}"

cd || exit 1
cd "$PROJECT_DIR" || { echo "❌ Project directory not found: $PROJECT_DIR"; exit 1; }
echo "📂 Current working directory: $(pwd)"
echo

VENV_DIR="venvs/unity_venv"
SCRIPTS_DIR="bash_scripts"

if [[ ! -d "$VENV_DIR/venv" ]]; then
  echo "🔧 Creating Unity venv..."
  bash "$SCRIPTS_DIR/make_venv.sh" "$VENV_DIR" python3.11
  echo "🔧 Installing requirements..."
  source "$VENV_DIR/venv/bin/activate"
  pip install -r requirements.txt
else
  echo "✅ Venv already exists. Activating..."
  source "$VENV_DIR/venv/bin/activate"
fi

# --- Load Manifest Row ---
mapfile -t LINES < "$MANIFEST"
HEADER="${LINES[0]}"
ROW="${LINES[$((SLURM_ARRAY_TASK_ID+1))]}"

echo
echo "==========================================="
echo "📄 Job Information"
echo "CSV Row ${SLURM_ARRAY_TASK_ID}: $ROW"
echo "Header: $HEADER"
echo "==========================================="
echo

IFS=',' read -ra COL_NAMES <<<"$HEADER"
IFS=',' read -ra COL_VALUES <<<"$ROW"

if [[ "${#COL_NAMES[@]}" -ne "${#COL_VALUES[@]}" ]]; then
  echo "❌ ERROR: Column mismatch!"
  echo "Header columns: ${#COL_NAMES[@]}"
  echo "Row columns: ${#COL_VALUES[@]}"
  exit 1
fi

# --- Build Python Args ---
declare -A EXCLUDE_MAP=(
  ["run_id"]=1
  ["partition"]=1
  ["cpus"]=1
  ["mem"]=1
  ["time"]=1
  ["gpus"]=1
  ["nodes"]=1
)

ARGS=""
RUN_ID=""
for i in "${!COL_NAMES[@]}"; do
  raw_key="${COL_NAMES[$i]}"
  raw_val="${COL_VALUES[$i]}"
  key="$(echo "$raw_key" | tr -d '\r\n' | xargs)"
  val="$(echo "$raw_val" | tr -d '\r\n' | xargs)"

  if [[ "$key" == "run_id" ]]; then
    RUN_ID="$val"
  fi

  if [[ -z "${EXCLUDE_MAP[$key]}" && -n "$key" && -n "$val" ]]; then
    ARGS+="--${key} ${val} "
  else
    if [[ -n "${EXCLUDE_MAP[$key]}" ]]; then
      echo "Skipping SLURM/internal key: $key"
    elif [[ -z "$val" ]]; then
      echo "Skipping empty value: $key"
    fi
  fi
done

ARGS="--run_id ${RUN_ID} ${ARGS}"

# --- Compose Job Name (if helpful for logs) ---
ALG="$(echo "$ARGS" | grep -oP '(?<=--allocation_method )\S+' || echo "sim")"
REQS="$(echo "$ARGS" | grep -oP '(?<=--num_requests )\S+' || echo "noreq")"
JOB_NAME="${ALG}_${REQS}_${RUN_ID}"

echo
echo "==========================================="
echo "🚀 Running Simulation"
echo "Job Name: $JOB_NAME"
echo "Python Command:"
echo "python run_sim.py ${ARGS}"
echo "==========================================="
echo

sleep 5
PY_OUT=$(python run_sim.py ${ARGS})
echo "$PY_OUT"

# --- Parse Output Path if available ---
RESULT_PATH=$(echo "$PY_OUT" | awk -F= '/^OUTPUT_DIR=/{print $2}')

echo
echo "==========================================="
echo "✅ Job Complete: ${SLURM_ARRAY_TASK_ID}"
if [[ -n "$RESULT_PATH" ]]; then
  echo "Mapped run_id=${RUN_ID} → ${RESULT_PATH}"
  echo "{\"run_id\":\"${RUN_ID}\",\"path\":\"${RESULT_PATH}\"}" >> "results/${JOB_DIR}/runs_index.json"
else
  echo "⚠️ No output path detected."
fi
echo "==========================================="
echo
