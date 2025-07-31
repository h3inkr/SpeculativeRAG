# Speculative RAG
<img width="1248" height="709" alt="image" src="https://github.com/user-attachments/assets/d020fdad-0e69-42c1-a95b-905cf913009d" />

Implementation for the paper [Enhancing Retrieval Augmented Generation through Drafting](https://arxiv.org/abs/2407.08223) (ICLR 2025)

## Installation
You can directly create a conda environment using the provided configuration file.
```bash
cd SpeculativeRAG
conda env create -f environment.yml
```

## Embedding & Retrieval
```bash
conda activate rag
cd SpeculativeRAG
bash ./scripts/embedding.sh
```

## Evaluation
Using the following script to evaluate Speculative RAG.
```bash
conda activate rag
cd SpeculativeRAG
bash ./scripts/inference.sh
```

## Fine-tuning (Option)
To fine-tune the RAG-drafter, first activate the environment and then run the script below. Be sure to update the input_file, model, and output_dir based on your setup.
```bash
conda activate rag
cd SpeculativeRAG
bash ./scripts/finetuning.sh
```
