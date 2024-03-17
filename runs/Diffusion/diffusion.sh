Model=$MODEL_HOME/codellama/CodeLlama-7b-Instruct

echo "Model=$Model"

torchrun --nproc_per_node 1 diffusion.py \
	--ckpt_dir $Model \
	--tokenizer_path $Model/tokenizer.model \
	--max_seq_len 512 --max_batch_size 4
