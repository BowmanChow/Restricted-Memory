#!/bin/bash
exp="baseline"
# exp="debug"
gpu_num="8"

# model="aott"
# model="aots"
# model="aotb"
# model="aotl"
model="r50_aotl"
# model="swinb_aotl"

source ~/.bashrc
conda activate aot
cd aot_plus

stage="pre_vost"
python tools/train.py --amp \
	--exp_name ${exp} \
	--stage ${stage} \
	--model ${model} \
	--gpu_num ${gpu_num} \
	--pretrained_path ../pretrained/R50_AOTL_PRE_YTB_DAV.pth \
	--batch_size 8 \
    --fix_random

# dataset="vost"
# split="val"
# python tools/eval.py --exp_name ${exp} --stage ${stage} --model ${model} \
# 	--dataset ${dataset} --split ${split} --gpu_num ${gpu_num} --ms 1.0 1.1 1.2 0.9 0.8