torchrun --nproc_per_node 1 $CODELLAMA_HOME/example_completion.py \
	--ckpt_dir $MODEL_HOME/codellama/CodeLlama-7b/jobnode.archive/2024-01-30 \
	--tokenizer_path $MODEL_HOME/codellama/CodeLlama-7b/jobnode.archive/2024-01-30/tokenizer.model \
	--max_seq_len 128 --max_batch_size 2
