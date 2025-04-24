#!/bin/bash
#SBATCH -p cpu
#SBATCH -c 1
#SBATCH --mem=32G
#SBATCH -t 2-00:00:00
#SBATCH --array=0-__N_JOBS__
#SBATCH -o __JOB_DIR__/slurm_%A_%a.out

module load python/3.11.7
source venvs/unity_venv/venv/bin/activate

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
