import numpy as np
import torch
torch.cuda.empty_cache()
import argparse
import faiss
import pickle
import re
from tqdm import tqdm
from itertools import islice
import logging
from math import ceil
import json

import sys
sys.path.append('./src/utils')
from multi_perspective import documents_to_clusters, multi_perspective_sampling
from rag_drafter import generate_draft
from rag_verifier import load_model_silently, compute_score
from metrics import batched_select_best_choice, batched_select_best_choice_open
from embedding import DPR, BasicRAG, SingleRAG
from utils import load_drafter_model
from prompt import PROMPT_TEMPLATES

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ["TORCH_USE_CUDA_DSA"] = '1'

# Set up logging
logging.basicConfig(
    filename='evaluation_log_56.txt',  # Log file name
    level=logging.INFO,  # Log level
    format='%(asctime)s - %(levelname)s - %(message)s'
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta_path", "-mp", type=str, required=True) 
    parser.add_argument("--model_name_or_path", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--retrieval_model_name_or_path", default="/mnt2/user4/coconut_documents/question_encoder")
    parser.add_argument("--embedding_path", default="/mnt2/user4/coconut_documents")
    parser.add_argument("--benchmark", "-b", required=True, choices=["arc", "csqa", "csqa2", "obqa", "piqa", "qasc", "wg"])
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_argument()
    rag = SingleRAG(args)
 
 # Load embedding
    with open(args.meta_path, "rb") as f:
        metadata = pickle.load(f)

    # Extract instruction-following QA pairs
    qa_pairs = {}
    for item in metadata:
        q = item.get("question")
        a = item.get("answer")
        choices = item.get("choices")
        docs = rag.retrieve(query=q, topk=10)
        if q and a:
            qa_pairs[q] = {
                "answer": a,
                "choices": choices, 
                "docs": docs
            }

    questions = []
    all_choices = []

    for question, others in tqdm(qa_pairs.items(), desc="QA pairs...", unit="pair"):
        q = question
        ch_texts = others["choices"]["text"]
        ch_labels = others["choices"]["label"]

        # inference_batch에서 사용하는 보기 형식 맞추기
        choices = []
        for label, text in zip(ch_labels, ch_texts):
            choices.append({
                "label": label,
                "text": text
            })

        questions.append(q)
        all_choices.append(choices)

    batch_size = 30  # OOM 
    outputs = []
    correct = 0
    total = 0
    num_batches = ceil(len(questions) / batch_size)

    for batch_idx in range(num_batches):
        start = batch_idx * batch_size
        end = start + batch_size

        batch_questions = questions[start:end]
        batch_choices = all_choices[start:end]
        batch_qa_pairs = list(qa_pairs.items())[start:end]

        batch_predictions, batch_docs, _, _ = rag.inference_batch(
            batch_questions,
            batch_choices,
            inst = PROMPT_TEMPLATES[args.benchmark]["inst"],
            reply = PROMPT_TEMPLATES[args.benchmark]["reply"]
        )

        for i, (question, others) in enumerate(batch_qa_pairs):
            pred = batch_predictions[i].strip()
            answer = others["answer"]
            is_correct = (pred == answer)

            correct += int(is_correct)
            total += 1

            outputs.append({
                "question": question,
                "ground_truth": answer,
                "generated_answer": pred,
                "is_correct": is_correct,
                "choices": others["choices"],
                "is_correct": is_correct,
                "retrieved_docs": batch_docs[i]
            })

    accuracy = correct / total
    print(f"✅ Accuracy: {accuracy:.2%}")

    output_file = f"results_{args.benchmark}.jsonl"
    with open(output_file, "w", encoding="utf-8") as f_out:
        for entry in outputs:
            json.dump(entry, f_out, ensure_ascii=False)
            f_out.write("\n")