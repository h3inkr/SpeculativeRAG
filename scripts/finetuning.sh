# finetune drafter
python ./src/finetune.py \
    --train_file ./data/train/knowledge_intensive/sft/sft_data_retrieved.jsonl \
    --model meta-llama/Llama-3.1-8B-Instruct \
    --output_dir ./test