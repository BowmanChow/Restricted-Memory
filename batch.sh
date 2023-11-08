#!/bin/bash
#SBATCH --mem=128g
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32    ## <- match to OMP_NUM_THREADS
#SBATCH --partition=gpuA40x4      ## <- or one of: cpu gpuA100x4 gpuA40x4 gpuA100x8 gpuMI100x8
#SBATCH --account=bbsh-delta-gpu
#SBATCH --job-name=vost
#SBATCH --time=12:00:00      ## hh:mm:ss for the job
### GPU options ###
#SBATCH --gpus-per-node=4
#SBATCH --mail-user=ziqip2@illinois.edu
#SBATCH -o ./delta_logs/baseline_r2.out
 
source ~/.bashrc
module reset
module load anaconda3_gpu
module unload anaconda3_gpu
conda activate /projects/bbsh/ziqip2/conda_envs/aot

echo "Running"
cd aot_plus
python tools/train.py --amp --exp_name baseline_r2 --stage pre_vost --model r50_aotl --gpu_num 4 --batch_size 8 \
    --fix_random 2 --pretrained_path ../pretrained_models/R50_AOTL_PRE_YTB_DAV.pth