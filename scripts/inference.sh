# Inference
# ARC-C
CUDA_VISIBLE_DEVICES=0 python src/inference.py \
    --meta_path ./data/inference/arc-c/test_meta.pkl \
    --drafter_path ./finetuned_drafter \
    --verifier meta-llama/Llama-3.1-8B-Instruct \
    --save_draft_path ./draft_arc-c.jsonl \
    --benchmark arc

# ARC-E
CUDA_VISIBLE_DEVICES=0 python src/inference.py \
    --meta_path ./data/inference/arc-e/test_514_meta.pkl \
    --drafter_path ./finetuned_drafter \
    --verifier meta-llama/Llama-3.1-8B-Instruct \
    --save_draft_path ./draft_arc-e.jsonl \
    --benchmark arc

# CSQA
CUDA_VISIBLE_DEVICES=0 python src/inference.py \
    --meta_path ./data/inference/csqa/dev_meta.pkl \
    --drafter_path ./finetuned_drafter \
    --verifier meta-llama/Llama-3.1-8B-Instruct \
    --save_draft_path ./draft_csqa.jsonl \
    --benchmark csqa

# CSQA2
CUDA_VISIBLE_DEVICES=0 python src/inference.py \
    --meta_path ./data/inference/csqa2/valid_meta.pkl \
    --drafter_path ./finetuned_drafter \
    --verifier meta-llama/Llama-3.1-8B-Instruct \
    --save_draft_path ./draft_csqa2.jsonl \
    --benchmark csqa2

# OBQA
CUDA_VISIBLE_DEVICES=0 python src/inference.py \
    --meta_path ./data/inference/obqa/test_512_meta.pkl \
    --drafter_path ./finetuned_drafter \
    --verifier meta-llama/Llama-3.1-8B-Instruct \
    --save_draft_path ./draft_obqa.jsonl \
    --benchmark obqa

# PIQA
CUDA_VISIBLE_DEVICES=0 python src/inference.py \
    --meta_path ./data/inference/piqa/valid_meta.pkl \
    --drafter_path ./finetuned_drafter \
    --verifier meta-llama/Llama-3.1-8B-Instruct \
    --save_draft_path ./draft_piqa.jsonl \
    --benchmark piqa

# QASC
CUDA_VISIBLE_DEVICES=0 python src/inference.py \
    --meta_path ./data/inference/qasc/valid_meta.pkl \
    --drafter_path ./finetuned_drafter \
    --verifier meta-llama/Llama-3.1-8B-Instruct \
    --save_draft_path ./draft_qasc.jsonl \
    --benchmark qasc

# WG
CUDA_VISIBLE_DEVICES=0 python src/inference.py \
    --meta_path ./data/inference/wg/dev_meta.pkl \
    --drafter_path ./finetuned_drafter \
    --verifier meta-llama/Llama-3.1-8B-Instruct \
    --save_draft_path ./draft_wg.jsonl \
    --benchmark wg