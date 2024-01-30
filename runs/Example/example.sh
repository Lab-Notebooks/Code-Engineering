export CUDA_VISIBLE_DEVICES=""

torchrun --nnodes 1 --nproc_per_node 1 $CODELLAMA_HOME/example_completion.py \
	--ckpt_dir $CODELLAMA_HOME/CodeLlama-7b/ --tokenizer_path $CODELLAMA_HOME/CodeLlama-7b/tokenizer.model \
	--max_seq_len 128 --max_batch_size 2
