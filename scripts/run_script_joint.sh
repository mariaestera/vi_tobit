#!/bin/bash
#SBATCH --job-name=tobit              # 1. Job name
##SBATCH --mail-type=BEGIN,END,FAIL   # 2. Send email upon events (Options: NONE, BEGIN, END, FAIL, ALL)
#SBATCH --partition=defaultp          # 3. Request a partition
#SBATCH --qos=normal                  # 4. Request a QoS
#SBATCH --ntasks=5                    # 5. Max 5 równoległych zadań na raz (rozmiar batcha)
#SBATCH --nodes=1                     #    Request number of node(s)
#SBATCH --mem=20G                     # 6. Request total amount of RAM (dzielone między równoległe procesy)
#SBATCH --time=0-02:00:00             # 7. Job execution duration limit day-hour:min:sec
##SBATCH --output=%x_%j.out           # 8. Standard output log as $job_name_$job_id.out
#SBATCH --output=%x.out
#SBATCH --error=%x.err
##SBATCH --error=%x_%j.err            #    Standard error log as $job_name_$job_id.err

# Do not export the local environment to the compute nodes
#    this is often needed because our cluster is quite heterogenous
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV

module load python/3.14.2
pip install arviz

date

scenario_name="scenario_1"
base_dir="Jupyter/vi_tobit"
input_folder_base="${base_dir}/simulations/X_design/${scenario_name}"
results_base="${base_dir}/simulations/results/${scenario_name}"
eval_folder="${base_dir}/simulations/eval/${scenario_name}"

mkdir -p "$results_base"
mkdir -p "$eval_folder"

seeds=(1 2 3 4 5)
batch_size=5

run_in_batches () {
    local -n cmds_ref=$1
    local n=${#cmds_ref[@]}
    local i=0
    while [ $i -lt $n ]; do
        local batch=("${cmds_ref[@]:$i:$batch_size}")
        for cmd in "${batch[@]}"; do
            eval "$cmd" &
        done
        wait
        i=$((i + batch_size))
    done
}

# =========================================================
# Pipeline per seed: symulacja -> Gibbs -> VI -> usuniecie X.npy
# =========================================================
pipeline_cmds=()
for seed in "${seeds[@]}"; do
    input_folder="${input_folder_base}__${seed}"
    vi_output_folder="${results_base}/vi__${seed}"
    gibbs_output_folder="${results_base}/gibbs__${seed}"

    mkdir -p "$input_folder"
    mkdir -p "$vi_output_folder"
    mkdir -p "$gibbs_output_folder"

    pipeline_cmds+=("
        set -e; \
        srun --exclusive -N1 -n1 python ${base_dir}/simulate_data_script.py \
            -n 1000 \
            -d 500 \
            -X_structure corr_blocks \
            --intercept true \
            --k 10 \
            --corr 0.7 \
            -l_perc 20 \
            -u_perc 80 \
            -snr 0.4 \
            --pi0 0.05 \
            --tau2 10 \
            --folder '${input_folder}' \
            --seed ${seed} && \
        srun --exclusive -N1 -n1 python ${base_dir}/gibbs_script.py \
            -input_folder '${input_folder}' \
            -output_folder '${gibbs_output_folder}' \
            -n_iter 1500 \
            -burn_in 1000 \
            --pi0 0.05 \
            --tau2 4 \
            --viz true \
            --seed ${seed} \
            --gamma_batch 10 && \
        srun --exclusive -N1 -n1 python ${base_dir}/vi_script.py \
            -input_folder '${input_folder}' \
            -output_folder '${vi_output_folder}' \
            --pi0 0.05 \
            --tau2 4 \
            --viz true \
            --seed ${seed} && \
        rm -f '${input_folder}/*'
    ")
done

echo "=== Pipeline: sim -> gibbs -> vi -> cleanup (per seed, batched) ==="
run_in_batches pipeline_cmds

# =========================================================
# Faza koncowa: evaluation (po usunieciu X.npy ze wszystkich folderow)
# =========================================================
echo "=== Evaluation ==="
srun --exclusive -N1 -n1 python ${base_dir}/evaluate_script.py \
    -data_folder "$input_folder_base" \
    -results_folder "$results_base" \
    -eval_folder "$eval_folder"

date