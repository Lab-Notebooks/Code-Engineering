torchrun --nproc_per_node 1 diffusion.py \
	--ckpt_dir $MODEL_HOME/codellama/CodeLlama-7b-Instruct/jobnode.archive/2024-01-30 \
	--tokenizer_path $MODEL_HOME/codellama/CodeLlama-7b-Instruct/jobnode.archive/2024-01-30/tokenizer.model \
	--max_seq_len 512 --max_batch_size 4
