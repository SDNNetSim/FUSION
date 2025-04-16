#!/bin/bash

#SBATCH -p cpu
#SBATCH -c 1
#SBATCH -G 0
#SBATCH --mem=32000
#SBATCH -t 0-7:00:00
#SBATCH --array=0-29  # 2 algorithms * 15 traffic volumes = 30 jobs
#SBATCH -o /dev/null

set -e

DEFAULT_DIR="/work/pi_vinod_vokkarane_uml_edu/git/FUSION/"

if [ -z "$1" ]; then
  echo "No directory provided. Using default directory: $DEFAULT_DIR"
  cd "$DEFAULT_DIR" || exit
else
  echo "Changing to user-specified directory: $1"
  cd "$1" || exit
fi
echo "Current directory: $(pwd)"

module load python/3.11.7

if [ ! -d "venvs/unity_venv/venv" ]; then
  ./bash_scripts/make_unity_venv.sh venvs/unity_venv python3.11
  pip install -r requirements.txt
  ./bash_scripts/register_rl_env.sh ppo SimEnv
  ./bash_scripts/register_rl_env.sh a2c SimEnv
fi

source venvs/unity_venv/venv/bin/activate

# Define algorithms and traffic volumes
algorithms=( "ppo" "a2c" )
traffic_volumes=( 50 100 150 200 250 300 350 400 450 500 550 600 650 700 750 )
erlang_step=50

network="NSFNet"
k_paths="3"

# Calculate indices based on SLURM_ARRAY_TASK_ID
alg_idx=$(( SLURM_ARRAY_TASK_ID / ${#traffic_volumes[@]} ))
traffic_idx=$(( SLURM_ARRAY_TASK_ID % ${#traffic_volumes[@]} ))

alg="${algorithms[$alg_idx]}"
erlang="${traffic_volumes[$traffic_idx]}"
erlang_stop=$(( erlang + erlang_step ))

# Create log directory
time_tag=$(date +%Y%m%d_%H%M%S)
log_dir="bash_scripts/slurm_logs/${alg}/$(date +%Y%m%d)/${network}/${erlang}"
mkdir -p "$log_dir"

exec > "${log_dir}/slurm_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_${SLURM_JOB_ID}_${time_tag}.out" 2>&1

echo "================================"
echo "SLURM Array ID    : $SLURM_ARRAY_TASK_ID"
echo "Algorithm         : $alg"
echo "Traffic Volume    : $erlang"
echo "Traffic Step      : $erlang_step"
echo "Erlang Stop       : $erlang_stop"
echo "Network           : $network"
echo "k_paths           : $k_paths"
echo "Log Directory     : $log_dir"
echo "Time Tag          : $time_tag"
echo "================================"

# Default parameters
params=(
  --erlang_start "$erlang"
  --erlang_stop "$erlang_stop"
  --network "$network"
  --k_paths "$k_paths"
  --path_algorithm "$alg"
  --max_iters 200
  --num_requests 5000
  --cores_per_link 3
  --reward 1
  --penalty -1
)

# Add algorithm-specific dynamic params
if [ "$alg" = "ppo" ]; then
  params+=(
    --alpha_start 0.00036
    --alpha_end 0.00001
    --decay_rate 0.4
    --epsilon_start 0.01
    --epsilon_end 0.0001
  )
elif [ "$alg" = "a2c" ]; then
  params+=(
    --alpha_start 0.0003
    --alpha_end 0.00001
    --decay_rate 0.4
    --epsilon_start 0.01
    --epsilon_end 0.0005
  )
fi

echo "Running command:"
echo "python run_rl_sim.py ${params[*]}"
echo "================================"

python run_rl_sim.py "${params[@]}" || echo "Error running Python script for job $SLURM_ARRAY_TASK_ID"
