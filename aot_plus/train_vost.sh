exp="aotplus"
# exp="debug"
gpu_num="1"
devices="4"

# model="aott"
# model="aots"
# model="aotb"
# model="aotl"
model="r50_aotl"
# model="swinb_aotl"
	
stage="pre_vost"
result_path=$(python -c "from tools.get_config import get_config ;cfg = get_config('$stage', '$exp', '$model') ;print(cfg.DIR_RESULT)")
echo "result_path=$result_path"
CUDA_VISIBLE_DEVICES=${devices} python tools/train.py --amp \
	--exp_name ${exp} \
	--stage ${stage} \
	--model ${model} \
	--gpu_num ${gpu_num} \
	--batch_size 2 \
	# --log ./debug_logs \
	# --fix


dataset="vost"
split="val"
eval_name="debug"
CUDA_VISIBLE_DEVICES=${devices} python tools/eval.py --result_path ${result_path} \
	--dataset ${dataset} --split ${split} --gpu_num 1 --ms 1.0 \
	--eval_name ${eval_name} \
	# --fix
result_path="${result_path}/eval/${dataset}/${eval_name}/"
echo "result_path=$result_path"


model_name=$(python -c "from configs.models.$model import ModelConfig ;print(ModelConfig().MODEL_NAME)")
cd ../evaluation
python ./evaluation_method.py --results_path ../aot_plus/${result_path}