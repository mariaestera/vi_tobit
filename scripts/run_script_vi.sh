#!/bin/bash
#SBATCH --job-name=tobit   # 1. Job name
##SBATCH --mail-type=BEGIN,END,FAIL    # 2. Send email upon events (Options: NONE, BEGIN, END, FAIL, ALL)
#SBATCH --partition=defaultp          # 3. Request a partition
#SBATCH --qos=normal                  # 4. Request a QoS
#SBATCH --ntasks=1                   # 5. Request total number of tasks (MPI workers)
#SBATCH --nodes=1                     #    Request number of node(s)
#SBATCH --mem=20G                     # 6. Request total amount of RAM
#SBATCH --time=0-00:30:00             # 7. Job execution duration limit day-hour:min:sec
##SBATCH --output=%x_%j.out            # 8. Standard output log as $job_name_$job_id.out
#SBATCH --output=%x.out
#SBATCH --error=%x.err
##SBATCH --error=%x_%j.err             #    Standard error log as $job_name_$job_id.err
# Do not export the local environment to the compute nodes
#    this is often needed because our cluster is quite heterogenous
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV
module load python/3.14.2
pip install arviz

# print the start time
date

scenario_name="scenario_1"
input_folder="Jupyter/vi_tobit/simulations/X_design/${scenario_name}"
mkdir -p "$input_folder"

srun python Jupyter/vi_tobit/simulate_data_script.py \
    -n 1000 \
    -d 500 \
    -X_structure corr_blocks \
    --k 10 \
    --corr 0.7 \
    -l_perc 20 \
    -u_perc 80 \
    -snr 0.4 \
    --pi0 0.05 \
    --tau2 10 \
    --folder "$input_folder" \
    --seed 42 \
    
output_folder="Jupyter/vi_tobit/simulations/results/vi_${scenario_name}_$(date +%Y%m%d-%H%M%S)"
mkdir -p "$output_folder"

srun python Jupyter/vi_tobit/mfvi_script.py \
    -input_folder "$input_folder"\
    -output_folder "$output_folder"\
    --pi0 0.05 \
    --tau2 10 \
    --seed 42 \
    --gamma_batch "-1"