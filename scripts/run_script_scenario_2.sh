#!/bin/bash
#SBATCH --job-name=tobit   # 1. Job name
##SBATCH --mail-type=BEGIN,END,FAIL    # 2. Send email upon events (Options: NONE, BEGIN, END, FAIL, ALL)
#SBATCH --partition=defaultp          # 3. Request a partition
#SBATCH --qos=normal                  # 4. Request a QoS
#SBATCH --ntasks=1                   # 5. Request total number of tasks (MPI workers)
#SBATCH --nodes=1                     #    Request number of node(s)
#SBATCH --mem=10G                     # 6. Request total amount of RAM
#SBATCH --time=0-01:00:00             # 7. Job execution duration limit day-hour:min:sec
##SBATCH --output=%x_%j.out            # 8. Standard output log as $job_name_$job_id.out
##SBATCH --error=%x_%j.err             #    Standard error log as $job_name_$job_id.err
# Do not export the local environment to the compute nodes
#    this is often needed because our cluster is quite heterogenous
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV
module load python/3.14.2
pip install arviz

# print the start time
date

seed=22
scenario_name="scenario_2"
input_folder="Jupyter/vi_tobit/simulations/X_design/${scenario_name}_${seed}"
output_folder="Jupyter/vi_tobit/simulations/${scenario_name}/${seed}"
mkdir -p "$input_folder"
mkdir -p "$output_folder"

srun python Jupyter/vi_tobit/simulate_data_script.py \
    -n 2000 \
    -d 1000 \
    -X_structure corr_blocks \
    --intercept 1 \
    --k 50 \
    --corr 0.9 \
    -l_perc 20 \
    -u_perc 80 \
    -snr 1 \
    --pi0 0.1 \
    --input_folder "$input_folder" \
    --output_folder "$output_folder" \
    --seed $seed \
    --test 1000 \

srun python Jupyter/vi_tobit/mfvi_blocks_script.py \
    -input_folder "$input_folder"\
    -output_folder "$output_folder"\
    --pi0 0.1 \
    --tau2 0.0050 \
    --seed $seed \
    --gamma_batch 1 \
    --em-warm_up 0 \
    --n_iter 1000

srun python Jupyter/vi_tobit/gibbs_script.py \
    -input_folder "$input_folder"\
    -output_folder "$output_folder"\
    -n_iter 2000 \
    -burn_in 1000 \
    --pi0 0.1 \
    --tau2 0.0050 \
    --seed $seed \
    --gamma_batch 1

srun python Jupyter/vi_tobit/ols_script.py \
    -input_folder "$input_folder"\
    -output_folder "$output_folder"\
    -n_iter 2000 \
    -burn_in 1000 \
    --pi0 0.1 \
    --tau2 0.0050 \
    --seed $seed \
    --gamma_batch 1