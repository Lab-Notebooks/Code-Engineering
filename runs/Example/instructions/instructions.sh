torchrun --nproc_per_node 1 $CODELLAMA_HOME/example_instructions.py \
	--ckpt_dir $MODEL_HOME/codellama/CodeLlama-7b-Instruct \
	--tokenizer_path $MODEL_HOME/codellama/CodeLlama-7b-Instruct/tokenizer.model \
	--max_seq_len 512 --max_batch_size 4
