import numpy as np
import torch
import argparse
import faiss
import pickle
import re
from tqdm import tqdm
from itertools import islice
import json
from math import ceil
import logging
import string

import sys
sys.path.append('./src/utils')
from embedding import DPR, BasicRAG, SingleRAG

# Set up logging
logging.basicConfig(
    filename='evaluation_arc-c.txt',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def parse_argument():
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl_path", "-jlp", type=str, required=True)
    parser.add_argument("--model_name_or_path", default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--retrieval_model_name_or_path", default="/mnt2/user4/coconut_documents/question_encoder")
    parser.add_argument("--embedding_path", default="/mnt2/user4/coconut_documents")
    parser.add_argument("--output_path", "-op", type=str, required=True)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_argument() 
    rag = SingleRAG(args)
    
    qa_pairs = []
    
    # ARC
    '''
    with open(args.jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            #if i >= 5:
            #    break  # 5개까지만 읽음
            item = json.loads(line)
            q = item.get("question")
            a = item.get("answerKey")
            if a == "1":
                a = "A"
            elif a == "2":
                a = "B"
            elif a == "3":
                a = "C"
            elif a == "4":
                a = "D"

            choices = item.get("choices", {})
            # label이 숫자인 경우 알파벳으로 변환
            if "label" in choices:
                new_labels = []
                for label in choices["label"]:
                    if label.isdigit():
                        idx = int(label)
                        if 0 <= idx < 26:
                            new_labels.append(string.ascii_uppercase[idx])  # A-Z
                        else:
                            new_labels.append(label)  # 범위 밖이면 그대로 둠
                    else:
                        new_labels.append(label)  # 이미 알파벳이면 그대로 둠
                choices["label"] = new_labels

            if q and a:
                qa_pairs.append({
                    "question": q,
                    "answer": a,
                    "choices": choices,
                    "qa_pairs": "null",
                    "ctxs": rag.retrieve(query=q, topk=3) 
                    #"ctxs": [] 
                })
    '''
    '''
    # CSQA, CSQA2
    with open(args.jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            #if i >= 5:
            #    break  # 5개까지만 읽음
            item = json.loads(line)
            base = item.get("question")
            q = base.get("stem")
            a = item.get("answerKey")
            choices = base.get("choices", [])
            choice_texts = [c.get("text", "") for c in choices]
            choice_labels = [c.get("label", "") for c in choices]

            new_choices = {
                "text": choice_texts,
                "label": choice_labels
            }

            if q and a:
                qa_pairs.append({
                    "question": q,
                    "answer": a,
                    "choices": new_choices,
                    "qa_pairs": "null",
                    "ctxs": rag.retrieve(query=q, topk=3) 
                    #"ctxs": [] 
                })
    '''
     
    # OBQA
    with open(args.jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            #if i >= 5:
            #    break  # 5개까지만 읽음
            item = json.loads(line)
            q = item.get("question_stem")
            a = item.get("answerKey")
            choices = item.get("choices", {})
            choice_texts = choices.get("text")
            choice_labels = choices.get("label")
            r = q +'\nChoices: '+'\n'.join(f"{label}. {text}" for label, text in zip(choices['label'], choices['text']))
            #print(r)

            if q and a:
                qa_pairs.append({
                    "question": q,
                    "answer": a,
                    "choices": choices,
                    "qa_pairs": "null",
                    "ctxs": rag.retrieve(query=r, topk=3) 
                    #"ctxs": [] 
                })
    
    '''
    # piqa
    with open(args.jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            #if i >= 5:
            #    break  # 5개까지만 읽음
            item = json.loads(line)
            q = item.get("goal")
            if item.get("answer") == "0":
                a = "A"
            else:
                a = "B"
            sol1 = item.get("sol1", {})
            sol2 = item.get("sol2", {})

            new_choices = {
                "text": [sol1, sol2],
                "label": ["A", "B"]
            }

            if q and a:
                qa_pairs.append({
                    "question": q,
                    "answer": a,
                    "choices": new_choices,
                    "qa_pairs": "null",
                    #"ctxs": rag.retrieve(query=q, topk=10) 
                    "ctxs": [] 
                })
    '''
    '''
    # qasc
    with open(args.jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            #if i >= 5:
            #    break  # 5개까지만 읽음
            item = json.loads(line)
            q = item.get("question")
            a = item.get("answerKey")
            choices = item.get("choices", {})
            choice_texts = choices.get("text")
            choice_labels = choices.get("label")

            if q and a:
                qa_pairs.append({
                    "question": q,
                    "answer": a,
                    "choices": choices,
                    "qa_pairs": "null",
                    #"ctxs": rag.retrieve(query=q, topk=10) 
                    "ctxs": [] 
                })
    '''
    '''
    # wg
    with open(args.jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            #if i >= 5:
            #    break  # 5개까지만 읽음
            item = json.loads(line)
            q = item.get("sentence")
            if item.get("answer") == "1":
                a = "A"
            else:
                a = "B"
            sol1 = item.get("option1", {})
            sol2 = item.get("option2", {})

            new_choices = {
                "text": [sol1, sol2],
                "label": ["A", "B"]
            }

            if q and a:
                qa_pairs.append({
                    "question": q,
                    "answer": a,
                    "choices": new_choices,
                    "qa_pairs": "null",
                    #"ctxs": rag.retrieve(query=q, topk=10) 
                    "ctxs": [] 
                })
        '''
    with open(args.output_path, "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=2)

    print(f"Number of QA pairs: {len(qa_pairs)}")
