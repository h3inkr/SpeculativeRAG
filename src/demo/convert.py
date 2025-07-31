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
import string

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
    
    qa_pairs = []

    with open(args.jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            item = json.loads(line)
            inst = "Given up to five answer candidates, A, B, C, D and E, choose the best answer choice.## Input:\n\n"
            q = item.get("question")
            a = item.get("answerKey")

            # answerKey가 숫자인 경우 알파벳으로 변환
            if a == "1":
                a = "A"
            elif a == "2":
                a = "B"
            elif a == "3":
                a = "C"
            elif a == "4":
                a = "D"

            choices = item.get("choices", {})
            labels = choices.get("label", [])
            texts = choices.get("text", [])

            # label이 숫자인 경우 알파벳으로 변환
            new_labels = []
            for label in labels:
                if label.isdigit():
                    idx = int(label)
                    if 0 <= idx < 26:
                        new_labels.append(string.ascii_uppercase[idx])  # A-Z
                    else:
                        new_labels.append(label)  # 범위 밖이면 그대로
                else:
                    new_labels.append(label)  # 이미 알파벳이면 그대로
            labels = new_labels

            # choices를 문자열로 변환 (A: xxx\nB: yyy ...)
            choice_str_list = []
            for label, text in zip(labels, texts):
                choice_str_list.append(f"{label}: {text}")
            choice_str = "\n".join(choice_str_list)

            id = item.get("id")

            if q and a:
                qa_pairs.append({
                    "instruction": inst + q + "\n" + choice_str,
                    "output": a,
                    "input": "",
                    "id": id,
                    "dataset_name": "arc"
                })
    
    with open(args.output_path, "w", encoding="utf-8") as f:
        for qa in qa_pairs:
            f.write(json.dumps(qa, ensure_ascii=False) + "\n")


    print(f"Number of QA pairs: {len(qa_pairs)}")