#!/bin/bash

#SBATCH --job-name=prokbiner
#SBATCH --nodes=1                    # 1 node
#SBATCH --ntasks-per-node=4         # 32 tasks per node
#SBATCH --time=24:00:00               
#SBATCH --gres=gpu:4                     # Use 1 GPU per job
#SBATCH --account=EUHPC_R04_194       # account name
#SBATCH --partition=boost_usr_prod # partition name
#SBATCH --qos=normal
#SBATCH --mem=100G                     # Memory per job

DATE=$(date +%Y_%m_%d)
# ERR_FILE="run_hyperparams_search_ia3${DATE}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err"
# OUT_FILE="run_hyperparams_search_ia3${DATE}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}${DATE}.out"
ERR_FILE="fine_tune_small_gtdb${DATE}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.err"
OUT_FILE="fine_tune_small_gtdb${DATE}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}${DATE}.out"
# Redirect logs manually
exec 2>$ERR_FILE
exec 1>$OUT_FILE

#set up right env
source /leonardo_scratch/fast/EUHPC_D16_066/sharedbins/UMAPprokbertbphageinpy311v4/miniconda3/etc/profile.d/conda.sh
conda activate /leonardo_scratch/fast/EUHPC_D16_066/sharedbins/UMAPprokbertbphageinpy311v4/miniconda3

export OMP_NUM_THREADS=1
srun torchrun --nnodes=1 --nproc_per_node=4 ./test_fintuning_gtdb.py