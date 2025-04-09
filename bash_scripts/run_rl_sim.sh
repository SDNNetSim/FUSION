#!/bin/bash

#SBATCH -p cpu
#SBATCH -c 1
#SBATCH -G 0
#SBATCH --mem=32000
#SBATCH -t 1-00:00:00
#SBATCH --array=0-7
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

algorithms=( "ppo" "a2c" )
traffic_volumes=( "200" "300" "500" "750" )

network="NSFNet"
k_paths="3"

alg_idx=$(( SLURM_ARRAY_TASK_ID / 4 ))      # each algorithm has 4 volumes
traffic_idx=$(( SLURM_ARRAY_TASK_ID % 4 ))  # remainder picks which volume

alg="${algorithms[$alg_idx]}"
erlang="${traffic_volumes[$traffic_idx]}"

time_tag=$(date +%Y%m%d_%H%M%S)
log_dir="bash_scripts/slurm_logs/${alg}/$(date +%Y%m%d)/${network}/${erlang}"
mkdir -p "$log_dir"

exec > "${log_dir}/slurm_${time_tag}.out" 2>&1

echo "================================"
echo "SLURM Array ID  : $SLURM_ARRAY_TASK_ID"
echo "Algorithm       : $alg"
echo "Traffic Volume  : $erlang"
echo "Network         : $network"
echo "k_paths         : $k_paths"
echo "Log Directory   : $log_dir"
echo "Time Tag        : $time_tag"
echo "================================"

python run_rl_sim.py \
  --erlang_start "$erlang" \
  --erlang_stop "$((erlang+100))" \
  --network "$network" \
  --k_paths "$k_paths" \
  --path_algorithm "$alg" \
|| echo "Error running Python script for job $SLURM_ARRAY_TASK_ID"
