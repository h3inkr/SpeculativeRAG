import numpy as np
import torch
torch.cuda.empty_cache()
import argparse
import faiss
import pickle
import re
from tqdm import tqdm
from itertools import islice
import json
from math import ceil

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

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def parse_argument():
    parser = argparse.ArgumentParser()
    #parser.add_argument("--index_path", "-ip", type=str, required=True, help="faiss index")
    parser.add_argument("--meta_path", "-mp", type=str, required=True) 
    parser.add_argument("--k", "-k", type=int, default=5) # draft based k documents, 2
    parser.add_argument("--m", "-m", type=int, default=2) # the number of drafts, 5
    parser.add_argument("--drafter", "-dr", type=str, default="mistralai/Mistral-7B-v0.1")
    parser.add_argument("--drafter_path", "-dp", type=str, default=None, help="Path to locally fine-tuned RAG drafter directory")
    parser.add_argument("--verifier", "-vr", type=str, default="mistralai/Mistral-7B-v0.1") # mistralai/Mistral-7B-v0.1 or mistralai/Mixtral-8x7B-v0.1
    parser.add_argument("--model_name_or_path", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--retrieval_model_name_or_path", default="/root/coconut_documents/question_encoder")
    parser.add_argument("--embedding_path", default="/root/coconut_documents")
    parser.add_argument("--save_draft_path", type=str, required=True, help="jsonl path")
    parser.add_argument("--benchmark", "-b", required=True, choices=["arc", "csqa", "csqa2", "obqa", "piqa", "qasc", "wg"])
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_argument()

    rag = SingleRAG(args)
 
    # Load embedding
    #index = faiss.read_index(args.index_path)
    with open(args.meta_path, "rb") as f:
        metadata = pickle.load(f)

    # Extract instruction-following QA pairs
    qa_pairs = {}
    #for item in metadata:
    for idx, item  in enumerate(metadata):
        if idx >= 100:
            break
        q = item.get("question")
        a = item.get("answer")
        choices = item.get("choices")
       # retrieved_docs = item.get("retrieved_docs")
        docs = rag.retrieve(query=q, topk=10)
        if q and a:
            qa_pairs[q] = {
                "answer": a,
                "choices": choices, 
                "docs": docs
            }

    # 추론할 질문 및 선택지 리스트 만들기
    questions = []
    all_choices = []

    for question, others in tqdm(qa_pairs.items(), desc="loading QA pairs...", unit="pair"):
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

    if args.drafter_path:
        drafter_path = args.drafter_path
    else:
        drafter_path = args.drafter

    drafter_model, drafter_tokenizer = load_drafter_model(drafter_path, device)
    verifier_model, verifier_tokenizer = load_model_silently(args.verifier, device)    

    best_responses = []
    with open(args.save_draft_path, "w", encoding="utf-8") as f_jsonl:
        for question, others in tqdm(qa_pairs.items(), desc="QA Evaluation", unit="pair"):
            answer = others["answer"]
            choices = others["choices"]
            docs = others["docs"]

            clusters = documents_to_clusters(docs=docs, m=args.m)
            subsets = multi_perspective_sampling(clusters=clusters, k=args.k)
    
            drafts = []
            generated_answer = ""
            for i, subset in enumerate(subsets):
                response, rationale, draft_score = generate_draft(
                    question=question, 
                    answer=answer, 
                    subset=subset, 
                    model=drafter_model,
                    tokenizer=drafter_tokenizer, 
                    device=device)
                drafts.append((response, rationale, draft_score))
                generated_answer += f"{response}"

            # drafts 단위로 jsonl에 기록: questions, choices, drafts(response, rationale, draft_score)
            json_line = {
                "question": question,
                "choices": choices,
                "drafts": drafts
            }
            f_jsonl.write(json.dumps(json_line, ensure_ascii=False) + "\n")

            scores = []
            for i, (response, rationale, draft_score) in enumerate(drafts):
                score = compute_score(
                    answer=response, 
                    rationale=rationale, 
                    draft_score=draft_score,
                    question =question,
                    device=device,
                    tokenizer=verifier_tokenizer,
                    model=verifier_model)
                scores.append(score)
            
            best_idx = np.argmax(scores)
            best_response = drafts[best_idx][0]
            best_rationale = drafts[best_idx][1]

            best_responses.append(best_response)

    batch_size = 30  # OOM 방지
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
        batch_responses = best_responses[start:end]
        
        #print(batch_responses)
        batch_predictions, _, _, _ = rag.inference_batch_speculative2(
            batch_questions,
            batch_choices,
            batch_responses,
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
                "choices": others["choices"],
                "is_correct": is_correct
            })

    accuracy = correct / total if total > 0 else 0
    print(f"✅ Accuracy: {accuracy:.2%}")