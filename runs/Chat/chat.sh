Model=$MODEL_HOME/codellama/CodeLlama-7b-Instruct

echo "Model=$Model"

torchrun --nproc_per_node 1 chat.py \
	--ckpt_dir $Model \
	--tokenizer_path $Model/tokenizer.model
