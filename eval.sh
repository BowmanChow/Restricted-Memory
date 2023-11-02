#!/bin/bash
#SBATCH --mem=128g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32    ## <- match to OMP_NUM_THREADS
#SBATCH --partition=cpu      ## <- or one of: cpu gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --account=bbsg-delta-cpu
#SBATCH --job-name=eval
#SBATCH --time=1:00:00      ## hh:mm:ss for the job
### GPU options ###
#SBATCH --mail-user=ziqip2@illinois.edu
#SBATCH -o ./delta_logs/pe_eval.out
 
source ~/.bashrc
module reset
conda activate /projects/bbsh/ziqip2/conda_envs/aot

cd evaluation

python evaluation_method.py --results_path ../aot_plus/results/pe_R50_AOTL/pre_vost/eval/vost/cap_1_3 \
    --dataset_path /scratch/bbsh/ziqip2/vos/VOST --re