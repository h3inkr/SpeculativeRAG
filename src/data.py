from typing import Dict, List, Callable, Tuple, Union, Callable
import logging
import os
import json
import re
import glob
import string
import spacy
from collections import Counter
from tqdm import tqdm
import numpy as np
from datasets import Dataset, load_dataset

logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger(__name__)

nlp = spacy.load("en_core_web_sm")

class BaseDataset:
    @classmethod
    def get_all_alias(cls, ground_truth_id: str) -> List[str]:
        return {}

    @classmethod
    def normalize_answer(cls, s):
        def remove_articles(text):
            return re.sub(r'\b(a|an|the)\b', ' ', text)
        def white_space_fix(text):
            return ' '.join(text.split())
        def remove_punc(text):
            exclude = set(string.punctuation)
            return ''.join(ch for ch in text if ch not in exclude)
        def lower(text):
            return text.lower()
        return white_space_fix(remove_articles(remove_punc(lower(s))))

    @classmethod
    def exact_match_score(
        cls,
        prediction: str,
        ground_truth: Union[str, List[str]],
        ground_truth_id: Union[str, List[str]] = None
    ):
        ground_truths = {ground_truth} if isinstance(ground_truth, str) else set(ground_truth)
        if ground_truth_id and isinstance(ground_truth_id, str):
            ground_truths.update(cls.get_all_alias(ground_truth_id))

        correct = np.max([int(cls.normalize_answer(prediction) == cls.normalize_answer(gt)) for gt in ground_truths])
        return {'correct': correct, 'incorrect': 1 - correct}

    @classmethod
    def f1_score(
        cls,
        prediction: str,
        ground_truth: Union[str, List[str]],
        ground_truth_id: Union[str, List[str]] = None
    ):
        ground_truths = {ground_truth} if isinstance(ground_truth, str) else set(ground_truth)
        if ground_truth_id and isinstance(ground_truth_id, str):
            ground_truths.update(cls.get_all_alias(ground_truth_id))
            
        final_metric = {'f1': 0, 'precision': 0, 'recall': 0}
        for ground_truth in ground_truths:
            normalized_prediction = cls.normalize_answer(prediction)
            normalized_ground_truth = cls.normalize_answer(ground_truth)
            if normalized_prediction in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
                continue
            if normalized_ground_truth in ['yes', 'no', 'noanswer'] and normalized_prediction != normalized_ground_truth:
                continue
            prediction_tokens = normalized_prediction.split()
            ground_truth_tokens = normalized_ground_truth.split()
            common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
            num_same = sum(common.values())
            if num_same == 0:
                continue

            precision = 1.0 * num_same / len(prediction_tokens)
            recall = 1.0 * num_same / len(ground_truth_tokens)
            f1 = (2 * precision * recall) / (precision + recall)
            for k in ['f1', 'precision', 'recall']:
                final_metric[k] = max(eval(k), final_metric[k])
        return final_metric


    def format(self, fewshot: int = 0):
        def _format(
            example: Dict,
            use_answer: bool = False,
            input_template_func: Callable = None,
        ):
            q = example['question']
            if 'cot' in example:
                cot = example['cot'] if type(example['cot']) is str else ''.join(example['cot'])
            else:
                cot = None
            a = example['answer']

            query = input_template_func(q)
            if use_answer:
                query += ('' if query[-1] in {'\n', ' '} else ' ') + self.output_template(cot, a)
            return query

        # demo
        demo = [{
            'question': self.examplars[i]['question'],
            'case': _format(self.examplars[i], use_answer=True, input_template_func=self.demo_input_template),
            'ctxs': self.examplars[i]['ctxs'] if 'ctxs' in self.examplars[i] else []
        } for i in range(fewshot)] if fewshot else []

        def _format_for_dataset(example):
            # case
            case = _format(example, use_answer=False, input_template_func=self.test_input_template)
            
            #if self.args.zeroshot:
            #case = [self.inst, self.reply, case, self.answer]
            #case = self.tokenizer.apply_chat_template(case)
        
            # ctx
            example['demo'] = demo
            example['case'] = case
            
            return example
        self.dataset = self.dataset.map(_format_for_dataset)
    
    def get_real_prediction(self, pred):
        return pred

class MatSciNLP(BaseDataset):
    examplars: List[Dict] = [
        ""
    ]
    
    demo_input_template = lambda self, ques: f'Question: {ques}\nAnswer:'
    #test_input_template = lambda self, ques: f'Following the examples above, answer the question by reasoning step-by-step.\n\nQuestion: {ques}\nAnswer:'
    #test_input_template = lambda self, ques: f'Following the examples above, answer the question.\n\nQuestion: {ques}\nAnswer:'
    test_input_template = lambda self, inst, ques: f"Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. \n\n ###Instruction : {inst}\n\n ###Input: {ques}\n\n ###Answer:"
    
    #output_template = lambda self, cot, ans: f'{cot} So the answer is {ans}.'
    
    #demo_input_template = lambda self, ques: f"{ques}"
    #test_input_template = lambda self, ques: f"{ques}"
    output_template = lambda self, ans: f'{ans}'


    def __init__(self, data_path: str):
        from utils import build_dataset
        
        #train_dataset, test_dataset = build_dataset(tasks = [0,1,2,3,4,5,6], explanation = True, setting = 'low_resource', train_size = 0.01, base_model = "meta-llama/Llama-2-7b-chat-hf", explanation_type = 'schema', even_split = False)
        train_dataset, test_dataset = build_dataset(tasks = [0,1,2,3,4,5,6], explanation = True, setting = 'low_resource', train_size = 0.01, base_model = "meta-llama/Llama-2-7b-chat-hf", explanation_type = 'llama', even_split = False)
        
        logger.info(f"Loading MatSciNLP from {data_path}")
        dataset = []
        #print(test_dataset)
        
        print(len(train_dataset), len(test_dataset))

        #for data in tqdm(test_dataset):
        for idx, data in tqdm(test_dataset.iterrows()):
            #print(data["texts"], data["questions"])
            #exit()
            
            #print(data)
            example = {
                "qid": idx, 
                "qtypes" : data["qtypes"],
                "inst" : data["questions"],
                "question": data["texts"],# + data["questions"], 
                #"ctxs" : data["texts"],
                #"cot": " ".join(data["facts"]), 
                "answer" : data["answers"]
            }
            dataset.append(example)
        self.dataset = Dataset.from_list(dataset)
        
        print(len(self.dataset))

    def get_real_prediction(self, pred):
        answer_prompts = ["the answer is"]
        for prmt in answer_prompts:
            if prmt in pred:
                beg = pred.find(prmt) + len(prmt) + 1
                pred = pred[beg:]
                return pred
        else:
            return ""

    def format(self, fewshot: int = 0):
        def _format(
            example: Dict,
            use_answer: bool = False,
            input_template_func: Callable = None,
        ):
            q = example['question']
            a = example['answer']
            inst = example["inst"]

            query = input_template_func(inst, q)
            if use_answer:
                query += ('' if query[-1] in {'\n', ' '} else ' ') + self.output_template(cot, a)
            return query

        # demo
        demo = [{
            'question': self.examplars[i]['question'],
            'case': _format(self.examplars[i], use_answer=True, input_template_func=self.demo_input_template),
            'ctxs': self.examplars[i]['ctxs'] if 'ctxs' in self.examplars[i] else []
        } for i in range(fewshot)] if fewshot else []

        def _format_for_dataset(example):
            # case
            case = _format(example, use_answer=False, input_template_func=self.test_input_template)
            # ctx
            example['demo'] = demo
            example['case'] = case
            return example
        self.dataset = self.dataset.map(_format_for_dataset)

    
    def __len__(self):
        return len(self.dataset)


class PIQA(BaseDataset):
    examplars: List[Dict] = [
    {
        "question":"How do you flood a room?\n Answer Choices: \n 1. fill it with objects. \n 2. fill it with water",
        "cot":"Too much water can cause flooding. Thus, if we want to flood a room, we should use water.",
        "answer" : "2"
    },
    {
        "question":"How can I get oil stains out of my driveway?\n Answer Choices: \n 1. Douse each stain with a couple cans of beer. \n 2. Douse each stain with a couple cans of soda.",
        "cot":"Sodium carbonate solution can wash away oil stains. The soda is a kind of sodium carbonate solution. Thus, you can use cans of soda to get oil stains out of your driveway.",
        "answer" : "2"
    },
    {
        "question":"Soothe a painful sunburn.\n Answer Choices: \n 1. Wait until brewed tea bag is cool, then apply on burn. \n 2. Wait until brewed tea bag is hot, then apply on burn.",
        "cot":"Sunburn can be alleviated by applying cold material. Thus, you should apply cool tea rather than hot tea bag to soothe your sunburn.",
        "answer" : "1"
    },
    {
        "question":"What can I use for fuel in an alcohol stove?\n Answer Choices: \n 1. Use acetone. \n 2. Use vinegar.",
        "cot":"Acetone is flammable, while vinegar is not. If you want to use something for fuel, the thing you use should be flammable. Thus, you should use acetone for fuel in an alcohol stove.",
        "answer" : "1"
    },
    {
        "question":"How can I cut the handles of metal cutlery?\n Answer Choices: \n 1. Use a hand saw to cut the handles. \n 2. Use a hand drill to cut the handles.",
        "cot":"A hand saw is used for making cuts and a hand drill is used for making holes. If you want to cut something, you should use a hand saw rather than hand drill.",
        "answer": "1"
    }
    ]
    

    
    
    def __init__(self, args, data_path: str):
        self.args = args
        logger.info(f"Loading PIQA from {data_path}")
        dataset = []
        with open("/home/user10/DRAGIN/data/piqa/dev.jsonl", "r") as f:
            test_dataset = f.readlines()
        with open("/home/user10/DRAGIN/data/piqa/dev-labels.lst", "r") as f:
            test_labels = f.readlines()    

        self.retrieve_examplars: List[Dict] = [
            {
                "question": "How can I get oil stains out of my driveway?\n Answer Choices: \n 1. Douse each stain with a couple cans of beer. \n 2. Douse each stain with a couple cans of soda.",
                "knowledge" : "1. The removal of oil stains from concrete surfaces \n 2. Substances can break down and remove oil \n 3. The property of beer and soda \n",
                "answer": "2"
            },
            {
                "question": "What can I use for fuel in an alcohol stove?\n Answer Choices: \n 1. Use acetone. \n 2. Use vinegar.",
                "knowledge": "1. The substances to use in an alchol stove \n 2. Combustion capabilities of acetone \n 3. Combustion capabilities of vinegar",
                "answer": "yes"
            },
            {
                "question": "Soothe a painful sunburn.\n Answer Choices: \n 1. Wait until brewed tea bag is cool, then apply on burn. \n 2. Wait until brewed tea bag is hot, then apply on burn.",
                "knowledge": "1. Treatments for sunburn \n 2. The effects of temperature on burned skin \n 3. How different temperatures affect damaged skin and the soothing properties",
                "answer": "yes"
            },
        ]
        self.labels = ["A","B"]
        self.demo_input_template = lambda ques: f'Question: {ques}\nAnswer:'

        #test_input_template = lambda self, ques: f'Following the examples above, answer the question by reasoning step-by-step.\n\nQuestion: {ques}\nAnswer:'
        
        if self.args.zeroshot:
            #self.test_input_template = lambda ques: f'You are an expert of world knowledge. I am going to ask you a question. Your response should be comprehensive and not contradicted with the following context if they are relevant. Otherwise, ignore them if they are not relevant.\n\nQuestion: {ques}\nAnswer:'
            #self.test_input_template = lambda ques: f'You are a helpful assistant for question answering. You are given a question or a text to complete and 2 possible solutions (labeled A and B). Your task is to choose the label corresponding to the best solution. \n\n Question: {ques} \n\n Answer:'
            self.inst = 'You are a helpful assistant for question answering. You are given a question or a text to complete and 2 possible solutions (labeled A and B). Your task is to choose the label corresponding to the best solution. \n\n'
            #self.test_input_template = lambda ques: f'Question: {ques} \n\nAnswer:'
            self.test_input_template = lambda ques: f'{ques}'
            self.reply = {'role': 'assistant', 'content': 'Yes, I understand. Please provide the question and the possible options.'}
            self.answer = {'role': 'assistant', 'content': 'Answer: '}
            self.knowledge_inst = "You are given a question or a text to complete and 2 possible solutions. Your task is to write one or more explanations that support the most likely solution. Note that: * there is always one solution that is correct and more likely than the others. * the explanations must support only the most likely solution and refute all the others. * the explanations must be simple and concise (max 15 words). Do you understand the task?"
            
    
        else:
            self.test_input_template = lambda ques: f'Following the examples above, answer the question by reasoning step-by-step.\n\nQuestion: {ques}\nAnswer:'

        self.output_template = lambda cot, ans: f'{cot} So the answer is {ans}.'


        if self.args.method == "zebra":
            self.inst =  """\
You are a helpful assistant for question answering. \
You are given a question or a text to complete, 2 possible solutions (labeled A and B), and a list of explanations. \
Your task is to choose the label corresponding to the best solution based on the given explanations. \
Do you understand the task?\
"""
            self.knowledge_inst = """\
You are given a question or a text to complete and 2 possible solutions. \
Your task is to write one or more explanations that support the most likely solution. \
Note that:
* there is always one solution that is correct and more likely than the others.
* the explanations must support only the most likely solution and refute all the others.
* the explanations must be simple and concise (max 15 words).
Do you understand the task?\
"""
            self.reply = 'Yes, I understand. Please provide the question and the possible options.'
            self.test_input_template = lambda ques: f'{ques}'
            self.answer = "Answer:"


        idx = 0
        for data, label in tqdm(zip(test_dataset, test_labels)):
            data = json.loads(data)
            #print(data)
            label = int(label)
            
            if label == 0:
                ans = "A"
            else:
                ans = "B"
            
            example = {
                "qid": idx, 
                "question": data["goal"] + "\n Options:" + "\n A." + data["sol1"] + "\n B." + data["sol2"], 
                #"ctxs" : data["texts"],
                #"cot": " ".join(data["facts"]), 
                "answer" : ans
            }
            idx += 1
            dataset.append(example)
        self.dataset = Dataset.from_list(dataset)
        
        print(len(self.dataset))

    def get_real_prediction(self, pred):
        answer_prompts = ["the answer is"]
        for prmt in answer_prompts:
            if prmt in pred:
                beg = pred.find(prmt) + len(prmt) + 1
                pred = pred[beg:] # delete final "."
                if pred.endswith("</s>"):
                    pred = pred[:len(pred) - len("</s>")]
                if pred.endswith("<|endoftext|>"):
                    pred = pred[:len(pred) - len("<|endoftext|>")]
                if pred.endswith("."):
                    pred = pred[:-1]
                
                #pred = pred[:1]
                return pred
        else:
            return ""

class CSQA(BaseDataset):
    examplars: List[Dict] = [
        {
            "question":"Google Maps and other highway and street GPS services have replaced what? \n A. atlas B. mexico C. countryside D. united states E. oceans ",
            "cot":"Electronic maps and GPS services are the modern version of paper atlas. In that case, the atlas have been replaced by Google Maps and other highway and street GPS services.",
            "answer" : "A"
        },
        {
            "question":"The fox walked from the city into the forest, what was it looking for? \n A. pretty flowers. B. hen house C. natural habitat D. storybook E. dense forest ",
            "cot":"Since the fox walk from the city into the forest, he may looks for something in the forest but not in the city. From all of the options, the natural habitat are usually away from cities.",
            "answer" : "C"
        },
        {
            "question":"You can share files with someone if you have a connection to a what? \n A. freeway B. radio C. wires D. computer network E. electrical circuit ",
            "cot":"Files usually can be stored in the computers. In that case, we can share them over the Internet. Thus, if we connect to a computer network, we can share the file with others. ",
            "answer" : "D"
        },
        {
            "question":"Too many people want exotic snakes. The demand is driving what to carry them? \n A. ditch B. shop C. north america D. outdoors E. pet shops ",
            "cot":"If people want exotic snakes, they may like to raise snakes as pets. If there is a demand for snakes as pets, pet shops will be pushed to carry them, in order to make more money. ",
            "answer" : "E"
        },
        {
            "question":"The body guard was good at his duties, he made the person who hired him what? \n A. better job B. feel safe C. irritated D. save money E. headache ",
            "cot":"The job of body guards is to ensure the safety and security of the employer. People ususally hire the body guard to make themselves safe. ",
            "answer" : "B"
        }
    ]
    

    #demo_input_template = lambda self, ques: f'Question: {ques}\nAnswer:'
    #test_input_template = lambda self, ques: f'Following the examples above, answer the question by reasoning step-by-step.\n\nQuestion: {ques}\nAnswer:'
    #output_template = lambda self, cot, ans: f'{cot} So the answer is {ans}.'

    def __init__(self, args, data_path: str):
        self.args = args
        logger.info(f"Loading CSQA from {data_path}")
        dataset = []
        with open("/home/user10/DRAGIN/data/csqa/commonsense_qa_test.json", "r") as f:
            #test_dataset = f.readlines()
            #dataset_1 = json.load(fz
            test_dataset = json.load(f)
        self.retrieve_examplars: List[Dict] = [
            {
                "question": "You can share files with someone if you have a connection to a what? \n A. freeway B. radio C. wires D. computer network E. electrical circuit ",
                "knowledge" : "1. how file-sharing works \n 2. the types of connections typically used for transferring data \n 3. the role of computer networks in facilitating file transfers \n",
                "answer": "D"
            },
            {
                "question": "Too many people want exotic snakes. The demand is driving what to carry them? \n A. ditch B. shop C. north america D. outdoors E. pet shops ",
                "knowledge": "1. where exotic snakes are typically sold \n 2. The place where exotic snakes can be sold",
                "answer": "E"
            },
            {
                "question": "Atlantic ocean is always bigger than indian ocean",
                "knowledge": "1. Fox behavior and their natural environment \n 2. What motivates a fox \n 3. What fox typically seeks in terms of habitat and resources",
                "answer": "C"
            },
        ]
        self.labels = ["A","B","C","D","E"]
        self.demo_input_template = lambda ques: f'Question: {ques}\nAnswer:'

        #test_input_template = lambda self, ques: f'Following the examples above, answer the question by reasoning step-by-step.\n\nQuestion: {ques}\nAnswer:'
        
        if self.args.zeroshot:
            #self.test_input_template = lambda ques: f'You are an expert of world knowledge. I am going to ask you a question. Your response should be comprehensive and not contradicted with the following context if they are relevant. Otherwise, ignore them if they are not relevant.\n\nQuestion: {ques}\nAnswer:'
            #self.test_input_template = lambda ques: f'You are a helpful assistant for question answering. You are given a question and 5 options (labeled A, B, C, D, and E). Your task is to choose the label corresponding to the best answer for the question. \n\n Question: {ques}\n\n Answer:'
            
            self.inst = 'You are a helpful assistant for question answering. You are given a question and 5 options (labeled A, B, C, D, and E). Your task is to choose the label corresponding to the best answer for the question. \n\n'
            #self.test_input_template = lambda ques: f'Question: {ques} \n\nAnswer:'
            self.test_input_template = lambda ques: f'{ques}'
            self.reply = {'role': 'assistant', 'content': 'Yes, I understand. Please provide the question and the possible options.'}
            self.answer = {'role': 'assistant', 'content': 'Answer: '}
            self.knowledge_inst = "You are given a question or a text to complete and 2 possible solutions. Your task is to write one or more explanations that support the most likely solution. Note that: * there is always one solution that is correct and more likely than the others. * the explanations must support only the most likely solution and refute all the others. * the explanations must be simple and concise (max 15 words). Do you understand the task?"
            
        else:
            self.test_input_template = lambda ques: f'Following the examples above, answer the question by reasoning step-by-step.\n\nQuestion: {ques}\nAnswer:'


        self.output_template = lambda cot, ans: f'{cot} So the answer is {ans}.'

        idx = 0
        for data in tqdm(test_dataset):
            #data = json.loads(data)
            #print(data)
            example = {
                "qid": idx, 
                "question": data["input"], 
                #"ctxs" : data["texts"],
                #"cot": " ".join(data["facts"]), 
                "answer" : data["answer"]
            }
            idx += 1
            dataset.append(example)
        self.dataset = Dataset.from_list(dataset)
        
        print(len(self.dataset))

    def get_real_prediction(self, pred):
        answer_prompts = ["the answer is"]
        for prmt in answer_prompts:
            if prmt in pred:
                beg = pred.find(prmt) + len(prmt) + 1
                pred = pred[beg:] # delete final "."
                if pred.endswith("</s>"):
                    pred = pred[:len(pred) - len("</s>")]
                if pred.endswith("<|endoftext|>"):
                    pred = pred[:len(pred) - len("<|endoftext|>")]
                if pred.endswith("."):
                    pred = pred[:-1]
                    
                return pred
        else:
            return ""

class CSQA2(BaseDataset):
    
    examplars: List[Dict] = [
    {
        "question": "A child cannot go to an R rated movie.",
        "cot" : "R-rated films are restricted to mature audiences due to content such as strong language, violence, or adult themes. Generally, viewers under 17 cannot attend without an accompanying parent or guardian.",
        "answer": "yes"
    },
    {
        "question": "Can a chicken be smaller than a baseball glove?",
        "cot": "A typical baseball glove varies in size but is generally around 10-12 inches in length. Baby chicks, when they hatch, are quite small. They can be just a few inches tall and weigh only a few ounces, making them easily smaller than a baseball glove.",
        "answer": "yes"
    },
    {
        "question": "Atlantic ocean is always bigger than indian ocean",
        "cot": "The Atlantic Ocean is the second-largest ocean in the world. The Indian Ocean is the third-largest ocean in the world.",
        "answer": "yes"
    },
    ]
    

    
    
    #demo_input_template = lambda self, ques: f'Question: {ques}\nAnswer:'
    #test_input_template = lambda self, ques: f'Following the examples above, answer the question by reasoning step-by-step.\n\nQuestion: {ques}\nAnswer:'
    #output_template = lambda self, cot, ans: f'{cot} So the answer is {ans}.'

    def __init__(self, args, data_path: str):
        self.args = args
        logger.info(f"Loading CSQA 2.0 from {data_path}")
        dataset = []
        with open("/home/user10/DRAGIN/data/csqa/CSQA2_dev.json", "r") as f:
            test_dataset = f.readlines()
            #dataset_1 = json.load(f)

        self.retrieve_examplars: List[Dict] = [
            {
                "question": "Would a pear sink in water?",
                "knowledge" : "1. The density of raw pear \n 2. The density of water \n 3. Relation between density and sinking \n",
                "answer": "yes"
            },
            {
                "question": "Can a chicken be smaller than a baseball glove?",
                "knowledge": "1. Size of chicken and glove \n 2. Volumn of chicken and glove \n 3. Height of chicken and glove",
                "answer": "yes"
            },
            {
                "question": "Atlantic ocean is always bigger than indian ocean",
                "knowledge": "1. Size of Atlantic ocean and Indian ocean \n 2. Volumn of water in Atlantic ocean and Indian ocean \n 3. Area of Atlantic ocean and Indian ocean",
                "answer": "yes"
            },
        ]
        '''
        
        self.retrieve_examplars: List[Dict] = [
            {
                "question": "A child cannot go to an R rated movie.",
                "knowledge" : "1. What is the age requirement for watching an R-rated movie? \n 2. What is an R-rated movie? \n 3. Why are movies given an R rating? \n",
                "answer": "yes" 
            },
            {
                "question": "Can a chicken be smaller than a baseball glove?",
                "knowledge": "1. What is the average size of a chicken? \n 2. What are the dimensions of a typical baseball glove? \n 3. How does the size of a chicken compare to a baseball glove?",
                "answer": "yes"
            },
            {
                "question": "Atlantic ocean is always bigger than indian ocean",
                "knowledge": "1. What is the total area of the Atlantic Ocean? \n 2. What is the total area of the Indian Ocean? \n 3. How do the Areas of the Atlantic and Indian Oceans differ?",
                "answer": "yes"
            },
        ]  
        '''
        
        #self.labels = ["yes","no"]
        
        self.labels = ["A","B"]
        self.demo_input_template = lambda ques: f'Question: {ques}\nAnswer:'
        
        
        
        #test_input_template = lambda self, ques: f'Following the examples above, answer the question by reasoning step-by-step.\n\nQuestion: {ques}\nAnswer:'
        
        if self.args.zeroshot:
            #self.test_input_template = lambda ques: f'You are an expert of world knowledge. I am going to ask you a question. Your response should be comprehensive and not contradicted with the following context if they are relevant. Otherwise, ignore them if they are not relevant.\n\nQuestion: {ques}\nAnswer:'
            #self.test_input_template = lambda ques: f'You are a helpful assistant for question answering. You are given a question and 2 options (labeled A and B). Your task is to choose the label corresponding to the best answer for the question.\n\nQuestion: {ques} \n\n Options: A. yes B. no \n\nAnswer:'

            self.inst = 'You are a helpful assistant for question answering. You are given a question and 2 options (labeled A and B). Your task is to choose the label corresponding to the best answer for the question.\n\n'
            #self.test_input_template = lambda ques: f'Question: {ques} \n\n Options: A. Yes B. No \n\nAnswer:'
            self.test_input_template = lambda ques: f'{ques}\n Options: A.Yes \n B.No'


            #self.test_input_template = lambda ques: f'You are a helpful assistant for question answering. You are given a question and 2 options (labeled A and B). Your task is to choose the label corresponding to the best answer for the question.\n\nQuestion: {ques} \n\n Options: A: yes B: no \n\nAnswer:'

            #self.inst = {'role': 'user', 'content': 'You are a helpful assistant for question answering. You are given a question, 2 options (labeled A and B), and a list of explanations. Your task is to choose the label corresponding to the best answer for the question based on the given explanations. Do you understand the task?'}
            self.reply = {'role': 'assistant', 'content': 'Yes, I understand. Please provide the question and the possible options.'}
            self.answer = {'role': 'assistant', 'content': 'Answer: '}
            #inst = {'role': 'user', 'content': 'You are a helpful assistant for question answering. You are given a question, 2 options (labeled A and B), and a list of explanations. Your task is to choose the label corresponding to the best answer for the question based on the given explanations. Do you understand the task?'}
            #reply = {'role': 'assistant', 'content': 'Yes, I understand. Please provide the question and the possible options.'}
            #question = {'role': 'user', 'content': 'Question:\nIs it true that some one tenth of the US population is also greater than the entire population of Norway?\n\nOptions:\n* A: Yes\n* B: No\n\nExplanations:\n'}
            
            #self.test_input_template = lambda ques: f"Question:\n{ques}\n\nOptions:\n* A: Yes\n* B: No"
            #self.test_input_template = lambda ques: "[{'role': 'user', 'content': 'You are a helpful assistant for question answering. You are given a question and 2 options (labeled A and B). Your task is to choose the label corresponding to the best answer for the question. Do you understand the task?'}, {'role': 'assistant', 'content': 'Yes, I understand. Please provide the question and the possible options.'}, {'role': 'user', 'content': 'Question:\n%s\n\nOptions:\n* A: Yes\n* B: No'}, {'role': 'assistant', 'content': 'Answer: '}]"%ques
            self.knowledge_inst = "You are given a question or a text to complete and 2 possible solutions. Your task is to write one or more explanations that support the most likely solution. Note that: * there is always one solution that is correct and more likely than the others. * the explanations must support only the most likely solution and refute all the others. * the explanations must be simple and concise (max 15 words). Do you understand the task?"
            
            
            pred = {'role': 'assistant', 'content': 'Answer: '}
        
            
        else:
            self.test_input_template = lambda ques: f'Following the examples above, answer the question by reasoning step-by-step.\n\nQuestion: {ques}\nAnswer:'


        self.output_template = lambda cot, ans: f'{cot} So the answer is {ans}.'


        idx = 0
        for data in tqdm(test_dataset):
            data = json.loads(data)
            #print(data)
            if data["answer"] == "yes":
                answer = "A"
            else:
                answer = "B"
            
            example = {
                "qid": data["id"],
                "question": data["question"], 
                #"ctxs" : data["texts"],
                #"cot": " ".join(data["facts"]), 
                "answer" : answer
            }
            idx += 1
            dataset.append(example)
        self.dataset = Dataset.from_list(dataset)
        
        print(len(self.dataset))

    def get_real_prediction(self, pred):
        answer_prompts = ["the answer is"]
        for prmt in answer_prompts:
            if prmt in pred:
                beg = pred.find(prmt) + len(prmt) + 1
                pred = pred[beg:]
                if pred[0:3].lower() == 'yes':
                    return "yes"
                else:
                    return "no"
        else:
            return ""

class StrategyQA(BaseDataset):
    examplars: List[Dict] = [
        {
            'question': 'Do hamsters provide food for any animals?',
            'ctxs': [(None, "Hamsters are prey animals."),
                (None, "Prey animals provide food for predators.")],
            'cot': ('Hamsters are prey animals. ',
                'Prey are food for predators. ',
                'Thus, hamsters provide food for some animals.'),
            'answer': 'yes',
        },
        {
            'question': 'Could Brooke Shields succeed at University of Pennsylvania?',
            'ctxs': [(None, "Brooke Shields graduated from Princeton University."),
                (None, "Princeton is ranked as the number 1 national college by US news."),
                (None, "University of Pennsylvania is ranked as number 6 national college by US news."),
                (None, "Princeton only admits around 6 percent of applicants as of 2018."),
                (None, "University of Pennsylvania accepts around 9% of applicants as of 2018.")],
            'cot': ('Brooke Shields went to Princeton University. ',
                'Princeton University is about as academically rigorous as the University of Pennsylvania. ',
                'Thus, Brooke Shields could also succeed at the University of Pennsylvania.'),
            'answer': 'yes',
        },
        {
            'question': "Hydrogen's atomic number squared exceeds number of Spice Girls?",
            'ctxs': [(None, "Hydrogen is the first element and has an atomic number of one."),
                (None, "The Spice Girls has five members."),
                (None, "To square a number, you multiply it by itself.")],
            'cot': ("Hydrogen has an atomic number of 1. ",
                "1 squared is 1. ",
                "There are 5 Spice Girls. ",
                "Thus, Hydrogen's atomic number squared is less than 5."),
            'answer': 'no',
        },
        {
            'question': "Is it common to see frost during some college commencements?",
            'ctxs': [(None, "Frost isn't uncommon to see during the month of December, as it is the winter."),
                (None, "College commencement ceremonies often happen during the months of December, May, and sometimes June.")],
            'cot': ("College commencement ceremonies can happen in December, May, and June. ",
                "December is in the winter, so there can be frost. ",
                "Thus, there could be frost at some commencements."),
            'answer': 'yes',
        },
        {
            'question': "Could a llama birth twice during War in Vietnam (1945-46)?",
            'ctxs': [(None, "The War in Vietnam (1945-46) lasted around 6 months."),
                (None, "The gestation period for a llama is 11 months.")],
            'cot': ("The War in Vietnam was 6 months. ",
                "The gestation period for a llama is 11 months, which is more than 6 months. ",
                "Thus, a llama could not give birth twice during the War in Vietnam."),
            'answer': 'no',
        },
        {
            'question': "Would a pear sink in water?",
            'ctxs': [(None, "The density of a raw pear is about 0.59 g/cm^3."),
                (None, "The density of water is about 1 g/cm^3."),
                (None, "Objects only sink if they are denser than the surrounding fluid.")],
            'cot': ("The density of a pear is about 0.6g/cm^3, which is less than water. ",
                "Objects less dense than water float. ",
                "Thus, a pear would float."),
            'answer': 'no',
        }
    ]

    #demo_input_template = lambda self, ques: f'Question: {ques}\nAnswer:'
    #test_input_template = lambda self, ques: f'Following the examples above, answer the question by reasoning step-by-step.\n\nQuestion: {ques}\nAnswer:'
    #output_template = lambda self, cot, ans: f'{cot} So the answer is {ans}.'

    def __init__(self, args, data_path: str):
        self.args = args
        logger.info(f"Loading StrategyQA from {data_path}")
        dataset = []
        
        
        self.demo_input_template = lambda ques: f'Question: {ques}\nAnswer:'
        
        if self.args.stepback:
            self.test_input_template = lambda ques: f'You are an expert of world knowledge. I am going to ask you a question. Your response should be comprehensive and not contradicted with the following context if they are relevant. Otherwise, ignore them if they are not relevant.\n\nQuestion: {ques}\nAnswer:'
        else:
            self.test_input_template = lambda ques: f'Following the examples above, answer the question by reasoning step-by-step.\n\nQuestion: {ques}\nAnswer:'


        #test_input_template = lambda self, ques: f'Following the examples above, answer the question by reasoning step-by-step.\n\nQuestion: {ques}\nAnswer:'
        self.output_template = lambda cot, ans: f'{cot} So the answer is {ans}.'

        with open(os.path.join(data_path, "strategyqa_train.json"), "r") as fin:
            dataset_1 = json.load(fin)
        with open(os.path.join(data_path, "strategyqa_train_paragraphs.json"), "r") as fin:
            dataset_2 = json.load(fin)
        for data in tqdm(dataset_1):
            example = {
                "qid": data["qid"], 
                "question": data["question"], 
                "cot": " ".join(data["facts"]), 
                "answer": "yes" if data["answer"] == True else "no", 
            }
            title = []
            ctxs = []
            for evi in data["evidence"][0]:
                if type(evi) == list:
                    for t in evi:
                        if type(t) == list:
                            title.extend(t)
                        else:
                            title.append(t)
                else:
                    title.append(evi)
            for tl in title:
                if tl == "operation" or tl == "no_evidence":
                    continue
                if tl in dataset_2:
                    ctxs.append(dataset_2[tl]["content"])
            example["ctxs"] = " ".join(ctxs)
            dataset.append(example)
        self.dataset = Dataset.from_list(dataset)

    def get_real_prediction(self, pred):
        answer_prompts = ["the answer is"]
        for prmt in answer_prompts:
            if prmt in pred:
                beg = pred.find(prmt) + len(prmt) + 1
                pred = pred[beg:]
                if pred[0:3].lower() == 'yes':
                    return "yes"
                else:
                    return "no"
        else:
            return ""


class WikiMultiHopQA(BaseDataset):
    examplars: List[Dict] = [
        {
            'question': "When did the director of film Hypocrite (Film) die?",
            'cot': "The film Hypocrite was directed by Miguel Morayta. Miguel Morayta died on 19 June 2013.",
            'answer': "19 June 2013",
        },
        {
            'question': "Are both Kurram Garhi and Trojkrsti located in the same country?",
            'cot': "Kurram Garhi is located in the country of Pakistan. Trojkrsti is located in the country of Republic of Macedonia. Thus, they are not in the same country.",
            'answer': "no",
        },
        {
            'question': "Do director of film Coolie No. 1 (1995 Film) and director of film The Sensational Trial have the same nationality?",
            'cot': "Coolie No. 1 (1995 film) was directed by David Dhawan. The Sensational Trial was directed by Karl Freund. David Dhawan's nationality is India. Karl Freund's nationality is Germany. Thus, they do not have the same nationality.",
            'answer': "no",
        },
        {
            'question': "Who is Boraqchin (Wife Of Ögedei)'s father-in-law?",
            'cot': "Boraqchin is married to Ögedei Khan. Ögedei Khan's father is Genghis Khan. Thus, Boraqchin's father-in-law is Genghis Khan.",
            'answer': "Genghis Khan",
        },
        {
            'question': "Who was born first out of Martin Hodge and Ivania Martinich?",
            'cot': "Martin Hodge was born on 4 February 1959. Ivania Martinich was born on 25 July 1995. Thus, Martin Hodge was born first.",
            'answer': "Martin Hodge",
        },
        {
            'question': "When did the director of film Laughter In Hell die?",
            'cot': "The film Laughter In Hell was directed by Edward L. Cahn. Edward L. Cahn died on August 25, 1963.",
            'answer': "August 25, 1963",
        },
        {
            'question': "Which film has the director died later, The Gal Who Took the West or Twenty Plus Two?",
            'cot': "The film Twenty Plus Two was directed by Joseph M. Newman. The Gal Who Took the West was directed by Frederick de Cordova. Joseph M. Newman died on January 23, 2006. Fred de Cordova died on September 15, 2001. Thus, the person to die later from the two is Twenty Plus Two.",
            'answer': "Twenty Plus Two",
        },
        {
            'question': "Who is the grandchild of Krishna Shah (Nepalese Royal)?",
            'cot': "Krishna Shah has a child named Rudra Shah. Rudra Shah has a child named Prithvipati Shah. Thus, Krishna Shah has a grandchild named Prithvipati Shah.",
            'answer': "Prithvipati Shah",
        }
    ]
    demo_input_template = test_input_template = lambda self, ques: f'Question: {ques}\nAnswer:'
    output_template = lambda self, cot, ans: f'{cot} So the answer is {ans}.'

    def __init__(self, data_path: str): 
        logger.info(f"Loading WikiMultiHopQA from {data_path}")
        dataset = []
        with open(os.path.join(data_path, 'dev.json'), 'r') as fin:
            js = json.load(fin)
            for example in tqdm(js):
                qid = example['_id']
                question = example['question']
                ans = example['answer']
                ans_id = example['answer_id']
                # ctxs = example['ctxs']
                dataset.append({
                    'qid': qid,
                    'question': question,
                    'answer': ans,
                    'answer_id': ans_id,
                    # 'ctxs': ctxs,
                })
        self.dataset = Dataset.from_list(dataset)
        self.init_id_aliases(data_path)
        
    @classmethod
    def init_id_aliases(cls, data_path):
        cls.id_alias: Dict[str, List[str]] = {}
        with open(os.path.join(data_path, 'id_aliases.json'), 'r') as fin:
            for l in fin:
                l = json.loads(l)
                cls.id_alias[l['Q_id']] = l['aliases']

    @classmethod
    def get_all_alias(cls, ground_truth_id: str) -> List[str]:
        if ground_truth_id and ground_truth_id in cls.id_alias:
            return cls.id_alias[ground_truth_id]
        else:
            return []

    def get_real_prediction(self, pred):
        if "the answer is" in pred:
            beg = pred.find("the answer is") + len("the answer is") + 1
            pred = pred[beg:] # delete final "."
            if pred.endswith("</s>"):
                pred = pred[:len(pred) - len("</s>")]
            if pred.endswith("<|endoftext|>"):
                pred = pred[:len(pred) - len("<|endoftext|>")]
            if pred.endswith("."):
                pred = pred[:-1]
            return pred
        else:
            return pred


class HotpotQA(BaseDataset):
    examplars: List[Dict] = [
        {
            'question': "Jeremy Theobald and Christopher Nolan share what profession?",
            'cot': "Jeremy Theobald is an actor and producer. Christopher Nolan is a director, producer, and screenwriter. Therefore, they both share the profession of being a producer.",
            'answer': "producer",
        },
        {
            'question': "What film directed by Brian Patrick Butler was inspired by a film directed by F.W. Murnau?",
            'cot': "Brian Patrick Butler directed the film The Phantom Hour. The Phantom Hour was inspired by the films such as Nosferatu and The Cabinet of Dr. Caligari. Of these Nosferatu was directed by F.W. Murnau.",
            'answer': "The Phantom Hour.",
        },
        {
            'question': "How many episodes were in the South Korean television series in which Ryu Hye-young played Bo-ra?",
            'cot': "The South Korean television series in which Ryu Hye-young played Bo-ra is Reply 1988. The number of episodes Reply 1988 has is 20.",
            'answer': "20",
        },
        {
            'question': "Were Lonny and Allure both founded in the 1990s?",
            'cot': "Lonny (magazine) was founded in 2009. Allure (magazine) was founded in 1991. Thus, of the two, only Allure was founded in 1990s.", 
            'answer': "no",
        },
        {
            'question': "Vertical Limit stars which actor who also played astronaut Alan Shepard in \"The Right Stuff\"?",
            'cot': "The actor who played astronaut Alan Shepard in \"The Right Stuff\" is Scott Glenn. The movie Vertical Limit also starred Scott Glenn.",
            'answer': "Scott Glenn",
        },
        {
            'question': "What was the 2014 population of the city where Lake Wales Medical Center is located?",
            'cot': "Lake Wales Medical Center is located in the city of Polk County, Florida. The population of Polk County in 2014 was 15,140.",
            'answer': "15,140",
        },
        {
            'question': "Who was born first? Jan de Bont or Raoul Walsh?",
            'cot': "Jan de Bont was born on 22 October 1943. Raoul Walsh was born on March 11, 1887. Thus, Raoul Walsh was born the first.",
            'answer': "Raoul Walsh",
        },
        {
            'question': "In what country was Lost Gravity manufactured?",
            'cot': "The Lost Gravity (roller coaster) was manufactured by Mack Rides. Mack Rides is a German company.",
            'answer': "Germany",
        },
        {
            'question': "Which of the following had a debut album entitled \"We Have an Emergency\": Hot Hot Heat or The Operation M.D.?",
            'cot': "The debut album of the band \"Hot Hot Heat\" was \"Make Up the Breakdown\". The debut album of the band \"The Operation M.D.\" was \"We Have an Emergency\".",
            'answer': "The Operation M.D.",
        },
        {
            'question': "How many awards did the \"A Girl Like Me\" singer win at the American Music Awards of 2012?",
            'cot': "The singer of \"A Girl Like Me\" singer is Rihanna. In the American Music Awards of 2012, Rihana won one award.",
            'answer': "one",
        },
        {
            'question': "The actor that stars as Joe Proctor on the series \"Power\" also played a character on \"Entourage\" that has what last name?",
            'cot': "The actor that stars as Joe Proctor on the series \"Power\" is Jerry Ferrara. Jerry Ferrara also played a character on Entourage named Turtle Assante. Thus, Turtle Assante's last name is Assante.",
            'answer': "Assante",
        },
    ]

    demo_input_template = lambda self, ques: f'Question: {ques}\nAnswer:'
    test_input_template = lambda self, ques: f'Answer the following question by reasoning step-by-step, following the example above.\nQuestion: {ques}\nAnswer:' 
    output_template = lambda self, cot, ans: f'{cot} So the answer is {ans}.'

    def __init__(self, data_path: str):
        logger.info(f"Loading HotpotQA from {data_path}")
        dataset = []
        with open(os.path.join(data_path, 'hotpotqa-dev.json'), "r") as fin:
            js = json.load(fin)
            for example in tqdm(js):
                qid = example["_id"]
                question = example["question"]
                answer = example['answer']
                context = example['context']
                dataset.append({
                    'qid': qid,
                    'question': question,
                    'answer': answer,
                    # 'ctxs': context,
                })
        self.dataset = Dataset.from_list(dataset)

    def get_real_prediction(self, pred):
        answer_prompts = ["the answer is"]
        for prmt in answer_prompts:
            if prmt in pred:
                beg = pred.find(prmt) + len(prmt) + 1
                pred = pred[beg:] # delete final "."
                if pred.endswith("</s>"):
                    pred = pred[:len(pred) - len("</s>")]
                if pred.endswith("<|endoftext|>"):
                    pred = pred[:len(pred) - len("<|endoftext|>")]
                if pred.endswith("."):
                    pred = pred[:-1]
                return pred
        else:
            return ""


class IIRC(BaseDataset):
    examplars: List[Dict] = [
        {
            "question": "What is the age difference between the kicker and the quarterback for the Chargers?",
            "cot": "The kicker for the Chargers is Nate Kaeding. The quarterback (QB) for the Chargers is Philip Rivers. Nate Kaeding was born in the year 1982. Philip Rivers was born in the year 1981. Thus, the age difference between them is of 1 year.",
            "answer": "1"
        },
        {
            "question": "How many years was the ship that took the battalion from New South Wales to Ceylon in service?",
            "cot": "The ship that took the battalion from New South Wales to Ceylon is General Hewitt. General Hewitt was launched in Calcutta in 1811. General Hewitt was sold for a hulk or to be broken up in 1864. So she served for a total of 1864 - 1811 = 53 years.",
            "answer": "53"
        },
        {
            "question": "What year was the theatre that held the 2016 NFL Draft built?",
            "cot": "The theatre that held the 2016 NFL Draft is Auditorium Theatre. The Auditorium Theatre was built in 1889.",
            "answer": "1889"
        },
        {
            "question": "How long had Milan been established by the year that Nava returned there as a reserve in the first team's defense?",
            "cot": "Nava returned to Milan as a reserve in the first team's defense in the year 1990. Milan had been established in the year 1899. Thus, Milan had been established for 1990 - 1899 = 91 years when Milan returned to Milan as a reserve in the first team's defense.",
            "answer": "91"
        },
        {
            "question": "When was the town Scott was born in founded?",
            "cot": "Scott was born in the town of Cooksville, Illinois. Cooksville was founded in the year 1882.",
            "answer": "1882"
        },
        {
            "question": "In what country did Wright leave the French privateers?",
            "cot": "Wright left the French privateers in Bluefield's river. Bluefields is the capital of the South Caribbean Autonomous Region (RAAS) in the country of Nicaragua.",
            "answer": "Nicaragua"
        },
        {
            "question": "Who plays the A-Team character that Dr. Hibbert fashioned his hair after?",
            "cot": "Dr. Hibbert fashioned his hair after Mr. T from The A-Team. Mr T.'s birthname is Lawrence Tureaud.",
            "answer": "Lawrence Tureaud"
        },
        {
            "question": "How many people attended the conference held near Berlin in January 1942?",
            "cot": "The conference held near Berlin in January 1942 is Wannsee Conference. Wannsee Conference was attended by 15 people.",
            "answer": "15"
        },
        {
            "question": "When did the country Ottwalt went into exile in founded?",
            "cot": "Ottwalt went into exile in the country of Denmark. Denmark has been inhabited since around 12,500 BC.",
            "answer": "12,500 BC"
        },
        {
            "question": "When was the J2 club Uki played for in 2001 founded?",
            "cot": "The J2 club that Uki played for is Montedio Yamagata. Montedio Yamagata was founded in 1984.",
            "answer": "1984"
        },
        {
            "question": "When was the person who produced A Little Ain't Enough born?",
            "cot": "A Little Ain't Enough was produced by Bob Rock. Bob Rock was born on April 19, 1954.",
            "answer": "April 19, 1954"
        },
        {
            "question": "Which of the schools Fiser is affiliated with was founded first?",
            "cot": "The schools that Fiser is affiliated with (1) Academy of Music, University of Zagreb (2) Mozarteum University of Salzburg (3) Croatian Music Institute orchestra. Academy of Music, University of Zagreb was founded in the year 1829. Mozarteum University of Salzburg was founded in the year 1841. Croatian Music Institute was founded in the year 1827. Thus, the school founded earliest of these is Croatian Music Institute.",
            "answer": "Croatian Music Institute"
        },
        {
            "question": "How many casualties were there at the battle that Dearing fought at under Jubal Early?",
            "cot": "Under Jubal Early, Dearing fought the First Battle of Bull Run. First Battle of Bull Run has 460 union casualties and 387 confederate casualties. Thus, in total the First Battle of Bull Run had 460 + 387 = 847 casualties.",
            "answer": "847"
        },
        {
            "question": "Which of the two congregations which provided leadership to the Pilgrims was founded first?",
            "cot": "The congregations which provided leadership to the Pilgrims are Brownists and Separatist Puritans. Brownist was founded in 1581. The Separatist Puritans was founded in 1640. Thus, Brownist was founded first.",
            "answer": "Brownist"
        },
        {
            "question": "How long had the Rock and Roll Hall of Fame been open when the band was inducted into it?",
            "cot": "The band was inducted into Rock and Roll Hall of Fame in the year 2017. Rock and Roll Hall of Fame was established in the year of 1983. Thus, Rock and Roll Hall of Fame been open for 2018 - 1983 = 34 years when the band was inducted into it.",
            "answer": "34"
        },
        {
            "question": "Did the Lord Sewer who was appointed at the 1509 coronation live longer than his king?",
            "cot": "Lord Sewer who was appointed at the 1509 coronation was Robert Radcliffe, 1st Earl of Sussex. Lord Sever's king in 1509 was Henry VIII of England. Robert Radcliffe, 1st Earl of Sussex was born in the year 1483, and died in the year 1542. So Robert lived for 1542 - 1483 = 59 years. Henry VIII of England was born in the year 1491 and died in the year 1547. So Henry VIII lived for 1547 - 1491 = 56 years. Thus, Robert Radcliffe lived longer than Henry VIII.",
            "answer": "yes"
        },
        {
            "question": "When was the place near where Manuchar was defeated by Qvarqvare established?",
            "cot": "Manuchar was defeated by Qvarqvare near Erzurum. Erzurum was founded during the Urartian period.",
            "answer": "Urartian period"
        },
        {
            "question": "What year was the man who implemented the 46 calendar reform born?",
            "cot": "The man who implemented the 46 calendar reform is Julius Caesar. Julius Caesar was born in the year 100 BC.",
            "answer": "100 BC"
        },
        {
            "question": "How many years after the first recorded Tommy John surgery did Scott Baker undergo his?",
            "cot": "The first recorded Tommy John surgery happened when it was invented in the year 1974. Scott Baker underwent Tommy John surgery in the year 2012. Thus, Scott Baker underwent Tommy John surgery 2012 - 1974 = 38 years after it was first recorded.",
            "answer": "38"
        },
        {
            "question": "Which was the older of the two players who found the net in the Double-Headed Eagle of the North in the sixth final for PAOK?",
            "cot": "The two players who found the net in the Double-Headed Eagle of the North in the sixth final for PAOK are Koudas and Matzourakis. Koudas was born on 23 November 1946. Matzourakis was born on 6 June 1949. Thus, the older person among the two is Koudas.",
            "answer": "Koudas"
        }
    ]

    demo_input_template = lambda self, ques: f'Question: {ques}\nAnswer:'
    test_input_template = lambda self, ques: f'Question: {ques}\nAnswer:' 
    output_template = lambda self, cot, ans: f'{cot} So the answer is {ans}.'

    def __init__(self, data_path: str):
        logger.info(f"Loading IIRC dev from {data_path}")
        dataset = []
        with open(os.path.join(data_path, 'dev.json'), "r") as fin:
            js = json.load(fin)
            for tmp in tqdm(js):
                for example in tmp['questions']:
                    qid = example["qid"]
                    question = example['question']

                    ans = example['answer']

                    if ans['type'] == 'none':
                        continue
                    elif ans['type'] == 'value' or ans['type'] == 'binary':
                        answer = [ans['answer_value']]
                    elif ans['type'] == 'span':
                        answer = [v['text'].strip() for v in ans['answer_spans']]
                    
                    # context = example['context']
                    dataset.append({
                        'qid': qid,
                        'question': question,
                        'answer': answer,
                        # 'ctxs': context,
                    })
        self.dataset = Dataset.from_list(dataset)

    def get_real_prediction(self, pred):
        answer_prompts = ["the answer is"]
        for prmt in answer_prompts:
            if prmt in pred:
                beg = pred.find(prmt) + len(prmt) + 1
                pred = pred[beg:] # delete final "."
                for stop_word in ["</s>", "<|endoftext|>", "\n", "."]:
                    if pred.endswith(stop_word):
                        pred = pred[:len(pred) - len(stop_word)]
                return pred
        else:
            return ""