

from dpr.models import init_biencoder_components
from dpr.options import set_cfg_params_from_state, setup_cfg_gpu
from dpr.utils.model_utils import (
    setup_for_distributed_mode,
    get_model_obj,
    load_states_from_checkpoint,
    move_to_device,
)
import yaml
from omegaconf import DictConfig, OmegaConf
import pickle
import tqdm
from datasets import load_dataset
import faiss
import time

from typing import Tuple, Union

import numpy as np
import torch
from transformers import PretrainedConfig
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from transformers.models.bert.modeling_bert import BertModel
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import List, Dict, Any

class GoldenRetrieverConfig(PretrainedConfig):
    model_type = "bert"

    def __init__(
        self,
        vocab_size=30522,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        hidden_act="gelu",
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        type_vocab_size=2,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=0,
        position_embedding_type="absolute",
        use_cache=True,
        classifier_dropout=None,
        projection_dim=None,
        **kwargs,
    ):
        super().__init__(pad_token_id=pad_token_id, **kwargs)

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.position_embedding_type = position_embedding_type
        self.use_cache = use_cache
        self.classifier_dropout = classifier_dropout
        self.projection_dim = projection_dim


class GoldenRetrieverModel(BertModel):
    config_class = GoldenRetrieverConfig

    def __init__(self, config, *args, **kwargs):
        super().__init__(config)
        self.layer_norm_layer = torch.nn.LayerNorm(
            config.hidden_size, eps=config.layer_norm_eps
        )
        self.projection: torch.nn.Module | None = None
        if config.projection_dim is not None:
            self.projection = torch.nn.Sequential(
                torch.nn.Linear(config.hidden_size, config.projection_dim),
                torch.nn.LayerNorm(config.projection_dim),
            )

    def forward(
        self, **kwargs
    ) -> Union[Tuple[torch.Tensor], BaseModelOutputWithPoolingAndCrossAttentions]:
        attention_mask = kwargs.get("attention_mask", None)
        model_outputs = super().forward(**kwargs)
        if attention_mask is None:
            pooler_output = model_outputs.pooler_output
        else:
            token_embeddings = model_outputs.last_hidden_state
            input_mask_expanded = (
                attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            )
            pooler_output = torch.sum(
                token_embeddings * input_mask_expanded, 1
            ) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

        pooler_output = self.layer_norm_layer(pooler_output)

        if self.projection is not None:
            pooler_output = self.projection(pooler_output)

        if not kwargs.get("return_dict", True):
            return (model_outputs[0], pooler_output) + model_outputs[2:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=model_outputs.last_hidden_state,
            pooler_output=pooler_output,
            past_key_values=model_outputs.past_key_values,
            hidden_states=model_outputs.hidden_states,
            attentions=model_outputs.attentions,
            cross_attentions=model_outputs.cross_attentions,
        )

class DPR:
    def __init__(
        self, 
        retrieval_model_name_or_path = "/mnt2/user4/coconut_documents/question_encoder",
        embedding_path = "/mnt2/user4/coconut_documents/",
        passage_file = None
    ):

        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model = GoldenRetrieverModel.from_pretrained(retrieval_model_name_or_path)
        self.embedding_path = embedding_path

        self.device = torch.device("cpu")
        
        self.model.to(self.device)
        self.model.eval()
        
        print("Building DPR indexes")

        self.p_reps = []

        #encode_file_path = sgpt_encode_file_path
        #dir_names = sorted(os.listdir(encode_file_path))

        # res = faiss.StandardGpuResources()
        # ngpus = faiss.get_num_gpus()

        #print("number of GPUs:", ngpus)
        #print(self.embedding_path)
        #self.faiss_index = faiss.IndexScalarQuantizer(768, faiss.ScalarQuantizer.QT_fp16, faiss.METRIC_INNER_PRODUCT)
        #self.faiss_index = faiss.index_cpu_to_gpu(res, 0, self.faiss_index)

        #self.faiss_index2 = faiss.IndexScalarQuantizer(768, faiss.ScalarQuantizer.QT_fp16, faiss.METRIC_INNER_PRODUCT)
        #self.faiss_index2 = faiss.index_cpu_to_gpu(res, 1, self.faiss_index2)
        self.faiss_index = faiss.IndexScalarQuantizer(768, faiss.ScalarQuantizer.QT_fp16, faiss.METRIC_INNER_PRODUCT)
            
        self.faiss_index2 = faiss.IndexScalarQuantizer(768, faiss.ScalarQuantizer.QT_fp16, faiss.METRIC_INNER_PRODUCT)
       

        split_parts = 3
        self.docs = []
        self.docs2 = []
        for i in tqdm(range(split_parts), ncols =100):
            st = time.time()

            f = open(self.embedding_path + "/explanation_embeddings%d/documents.jsonl"%(i+1), "r")

            docs = f.readlines()
            print(json.loads(docs[0]))
            docs = [json.loads(s)["text"] for s in docs]
            
            tp = torch.load(self.embedding_path + "/explanation_embeddings%d/embeddings.pt"%(i+1), map_location="cpu")    
            
            print("Embeddings : ", tp.shape)
            
            tp = torch.nn.functional.normalize(tp, p=2, dim=1)
            tp = tp.numpy().astype(np.float32)
            print("Spended_time:" , time.time() - st)
            
            if i == 2:
                self.faiss_index2.train(tp)
                self.faiss_index2.add(tp)
                self.docs2 += docs
            else:
                self.faiss_index.train(tp)            
                self.faiss_index.add(tp)
                self.docs += docs
            
            print(self.faiss_index.ntotal, self.faiss_index2.ntotal)
            
            ft = time.time()
            print("Spended_time:" , ft - st)

        print(len(self.docs), len(self.docs2))
        self.doc_length = len(self.docs)
        self.docs += self.docs2
        print(len(self.docs))

    def tokenize_with_specb(self, texts, is_query):
        # Tokenize without padding
        batch_tokens = self.tokenizer(texts, padding=False, truncation=True)   
        
        # Add padding
        batch_tokens = self.tokenizer.pad(batch_tokens, padding=True, return_tensors="pt")
        batch_tokens = {k: v.to(self.device) for k,v in batch_tokens.items()}
        return batch_tokens

    def get_weightedmean_embedding(self, batch_tokens):
        # Get the embeddings
        with torch.no_grad():
            # Get hidden state of shape [bs, seq_len, hid_dim]
            
            #last_hidden_state = self.model(**batch_tokens, output_hidden_states=True, return_dict=True).last_hidden_state
            last_hidden_state, _, _ = self.model(**batch_tokens)


        # Get weights of shape [bs, seq_len, hid_dim]
        weights = (
            torch.arange(start=1, end=last_hidden_state.shape[1] + 1)
            .unsqueeze(0)
            .unsqueeze(-1)
            .expand(last_hidden_state.size())
            .float().to(last_hidden_state.device)
        )

        # Get attn mask of shape [bs, seq_len, hid_dim]
        input_mask_expanded = (
            batch_tokens["attention_mask"]
            .unsqueeze(-1)
            .expand(last_hidden_state.size())
            .float()
        )

        # Perform weighted mean pooling across seq_len: bs, seq_len, hidden_dim -> bs, hidden_dim
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded * weights, dim=1)
        sum_mask = torch.sum(input_mask_expanded * weights, dim=1)

        embeddings = sum_embeddings / sum_mask

        return embeddings

    def retrieve(
        self, 
        queries: List[str], 
        topk: int = 1,
    ):
        #print(queries)
        batch_tokens = self.tokenize_with_specb(queries, is_query=True)
        
        #print(batch_tokens)
        
        with torch.no_grad():
            #_ , q_reps, _ = self.model(**batch_tokens)
            q_reps = self.model(**batch_tokens).pooler_output
        
        
        #print(q_reps)

        q_reps.detach()
        q_reps = torch.nn.functional.normalize(q_reps, p=2, dim=1)

        Distance, Index = self.faiss_index.search(q_reps.cpu().numpy(), k = topk * 10)
        Distance2, Index2 = self.faiss_index2.search(q_reps.cpu().numpy(), k = topk * 5)

        Index2 = Index2 + self.doc_length
        
        SuperIndex = np.concatenate([Index,Index2], axis = 1)
        SuperDistance = np.concatenate([Distance, Distance2], axis = 1)
        top_indices = np.argsort(SuperDistance, axis = 1)
        top_indices = top_indices[0][-topk:]
        Index = [SuperIndex[0][top_indices]]

        psgs = []
        for qid in range(q_reps.shape[0]):
            ret = []
            for j in range(topk):
                #idx = global_topk_indices[j][qid].item()
                #fid, rk = idx // topk, idx % topk
                psg = self.docs[Index[qid][j]]
                #print(psg)
                ret.append(psg)
            psgs.append(ret)
        return psgs


        
#  Successfully uninstalled datasets-2.14.1 예전 버전 - dragin
class DPR_ZEBRA:
    def __init__(
        self, 
        model_name_or_path,
        sgpt_encode_file_path,
        passage_file,
        augmentation = False,
        datasets = None,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model = GoldenRetrieverModel.from_pretrained("sapienzanlp/zebra-retriever-e5-base-v2")
        print(self.model)
        
        self.model.cuda(0)
        self.model.eval()

        print("Building DPR indexes")

        self.p_reps = []

        encode_file_path = sgpt_encode_file_path
        dir_names = sorted(os.listdir(encode_file_path))
        
        
        res = faiss.StandardGpuResources()
        self.faiss_index = faiss.IndexFlatL2(768)
        ngpus = faiss.get_num_gpus()

        print("number of GPUs:", ngpus)
        self.faiss_index = faiss.IndexScalarQuantizer(768, faiss.ScalarQuantizer.QT_fp16, faiss.METRIC_L2)
        self.faiss_index = faiss.index_cpu_to_gpu(res, 0, self.faiss_index)
        
        question_to_explanations = dict()
        explanations = load_dataset("sapienzanlp/zebra-kb-explanations", "all")["train"]
        for sample in explanations:
            question_id = sample["id"]
            question_to_explanations[question_id] = sample["positives"]
        
        with open("/home/user10/DRAGIN/data/documents.jsonl", "r") as f:
            x = f.readlines()
        
        print(len(question_to_explanations), len(x))
        
        if datasets != None:
            with open(DATA_PATH[datasets], "r") as f:
                train_data = f.readlines()
             
            id_dict = dict()   
            #id_dict = {s["id"]: len(id_dict) for s in train_data}
            for s in train_data:
                s = json.loads(s[:-1])
                id_dict[s["id"]] = len(id_dict)
        
        indices = []

        self.docs = []
        skip_num = 0
        for idx, s in enumerate(x):
            s = json.loads(s[:-1])
            question_id = s["id"]
            
            if datasets != None:
                if question_id in id_dict:
                    indices.append(idx)
                    skip_num += 1
                    continue
            
            if question_id not in question_to_explanations:
                skip_num += 1
                #print(question_id)
                s["explanation"] = ""
            else:
                s["explanation"] = question_to_explanations[question_id]
                
            #print(s)
            explanations = s["explanation"]
            #explanations = ""
            #for e in s["explanation"]:
            #    explanations += f"* {e}"
            
            
            #s["text"] = s["text"].replace("[SEP]","")
            
            example_question = s["text"].split(" [SEP] ")[0]
            example_choices = s["text"].split(" [SEP] ")[1:]
            example_choices = [
                {
                    "label": choice.split(".")[0].strip(),
                    "text": choice.split(".")[1].strip(),
                }
                for choice in example_choices
            ]

            #self.docs.append("Question: " + s["text"] + "\n\n" + "Explanations: " + explanations)
            #self.docs.append([s["text"], explanations])
            self.docs.append([example_question, example_choices, explanations])
            #self.docs.append(explanations)
            
        print(len(self.docs), len(question_to_explanations), skip_num)
        
        embeddings = torch.load("/home/user10/DRAGIN/data/embeddings.pt", map_location="cpu")
        print("Embeddings : ", embeddings.shape)
        #embeddings = embeddings.cpu().numpy().astype(np.float32)
        embeddings = embeddings.cpu().tolist()
        
        temp = []
        #for idx in indices:
            
        for idx in range(len(embeddings)):
            if idx in indices:
                continue
            temp.append(embeddings[idx][:])
        temp = np.array(temp).astype(np.float32)
        embeddings = temp
        
        
        print(len(self.docs), "Embeddings : ", embeddings.shape)
        
        
        if augmentation:
            #with open("/home/user10/DRAGIN/data/generated_knowledge/csqa2_1000samples.jsonl", "r") as f:
            #with open("/home/user10/DRAGIN/data/generated_knowledge/csqa2_samples_llama_70B_ver2.jsonl") as f:
            with open("/home/user10/DRAGIN/data/generated_knowledge/zebra_samples_llama_70B_ver2.jsonl", "r") as f:
                x = f.readlines()
                
            temp_embeddings = []
                
            temp_text = []
            for s in x[:-2]:
                #print(s)
                s = json.loads(s[:-1])
                knowledge = s["knowledge"]
                for k in knowledge:
                    #temp_text.append("Question:" + k)
                    temp_text.append(k)

            batch_size = 16
            for i in tqdm.tqdm(range(len(temp_text) // batch_size + 1), ncols = 100):
                text = temp_text[i * batch_size : (i+1) * batch_size]
                inputs = self.tokenizer(text, return_tensors = "pt", padding = True)
                attention_mask = inputs["attention_mask"].cuda()
                input_ids = inputs["input_ids"].cuda()
                
                emb = self.model(input_ids = input_ids, attention_mask = attention_mask).pooler_output
                
                temp_embeddings += emb.cpu().tolist()

            temp_embeddings = np.array(temp_embeddings).astype(np.float32)
            embeddings = np.concatenate([embeddings, temp_embeddings], axis = 0)
            self.docs = self.docs + temp_text

            print("Augmented: ", len(self.docs), "Embedding shapes :", embeddings.shape)
        
        self.faiss_index.train(embeddings)  
        self.faiss_index.add(embeddings)
        print(self.faiss_index.ntotal)
        
    def tokenize_with_specb(self, texts, is_query):
        # Tokenize without padding
        batch_tokens = self.tokenizer(texts, padding=False, truncation=True)   
        
        # Add padding
        batch_tokens = self.tokenizer.pad(batch_tokens, padding=True, return_tensors="pt")
        batch_tokens = {k: v.cuda() for k,v in batch_tokens.items()}
        return batch_tokens

    def retrieve(
        self, 
        queries: List[str], 
        topk: int = 1,
    ):
        #print(queries)
        batch_tokens = self.tokenize_with_specb(queries, is_query=True)
        
        with torch.no_grad():
            q_reps = self.model(**batch_tokens).pooler_output
        
        q_reps.detach()
        q_reps_trans = torch.transpose(q_reps, 0, 1)

        topk_values_list = []
        topk_indices_list = []
        
        topk_indices_list_custom = []
        
        Distance, Index = self.faiss_index.search(q_reps.cpu().numpy(), k = topk)

        psgs = []
        for qid in range(q_reps.shape[0]):
            ret = []
            for j in range(topk):
                psg = self.docs[Index[qid][j]]
                ret.append(psg)
            psgs.append(ret)
        return psgs


class DPR_ZEBRA_COCONUT:
    def __init__(
        self, 
        model_name_or_path,
        sgpt_encode_file_path,
        passage_file,
        augmentation = False,
        datasets = None,
    ):
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.model = GoldenRetrieverModel.from_pretrained("sapienzanlp/zebra-retriever-e5-base-v2")
        print(self.model)
        
        self.device = torch.device("cuda:0")
        
        self.model.to(self.device)
        
        
        self.model.eval()

        print("Building DPR indexes")

        self.p_reps = []

        encode_file_path = sgpt_encode_file_path
        dir_names = sorted(os.listdir(encode_file_path))
        
        
        res = faiss.StandardGpuResources()
        self.faiss_index = faiss.IndexFlatL2(768)
        ngpus = faiss.get_num_gpus()

        print("number of GPUs:", ngpus)
        self.faiss_index = faiss.IndexScalarQuantizer(768, faiss.ScalarQuantizer.QT_fp16, faiss.METRIC_L2)
        self.faiss_index = faiss.index_cpu_to_gpu(res, 0, self.faiss_index)
        

        with open("/home/user10/DRAGIN/data/explanation_embeddings_coconut_zebra/documents.jsonl", "r") as f:
            x = f.readlines()
        with open("/home/user10/DRAGIN/data/genkb.jsonl", "r") as f:
            y = f.readlines()

        self.docs = []
        skip_num = 0
        for idx, s in enumerate(x):
            s = json.loads(s[:-1])
            
            s2 = json.loads(y[idx][:-1])
            
            question_id = s["id"]
            
            #print(s)
            explanations = s2["text"]
            #explanations = ""
            #for e in s["explanation"]:
            #    explanations += f"* {e}"
            
            
            #s["text"] = s["text"].replace("[SEP]","")
            
            example_question = s["text"].split(" [SEP] ")[0]
            example_choices = s["text"].split(" [SEP] ")[1:]
            example_choices = [
                {
                    "label": choice.split(".")[0].strip(),
                    "text": choice.split(".")[1].strip(),
                }
                for choice in example_choices
            ]

            #self.docs.append("Question: " + s["text"] + "\n\n" + "Explanations: " + explanations)
            #self.docs.append([s["text"], explanations])
            self.docs.append([example_question, example_choices, explanations])
            #self.docs.append(explanations)
            
        print(len(self.docs))
        
        embeddings = torch.load("/home/user10/DRAGIN/data/explanation_embeddings_coconut_zebra/embeddings.pt", map_location="cpu")
        print("Embeddings : ", embeddings.shape)
        embeddings = embeddings.cpu().numpy().astype(np.float32)
        
        self.faiss_index.train(embeddings)  
        self.faiss_index.add(embeddings)
        print(self.faiss_index.ntotal)
        
    def tokenize_with_specb(self, texts, is_query):
        # Tokenize without padding
        batch_tokens = self.tokenizer(texts, padding=False, truncation=True)   
        
        # Add padding
        batch_tokens = self.tokenizer.pad(batch_tokens, padding=True, return_tensors="pt")
        batch_tokens = {k: v.to(self.device) for k,v in batch_tokens.items()}
        return batch_tokens

    def retrieve(
        self, 
        queries: List[str], 
        topk: int = 1,
    ):
        #print(queries)
        batch_tokens = self.tokenize_with_specb(queries, is_query=True)
        
        with torch.no_grad():
            q_reps = self.model(**batch_tokens).pooler_output
        
        q_reps.detach()
        q_reps_trans = torch.transpose(q_reps, 0, 1)

        topk_values_list = []
        topk_indices_list = []
        
        topk_indices_list_custom = []
        
        Distance, Index = self.faiss_index.search(q_reps.cpu().numpy(), k = topk)

        psgs = []
        for qid in range(q_reps.shape[0]):
            ret = []
            for j in range(topk):
                psg = self.docs[Index[qid][j]]
                ret.append(psg)
            psgs.append(ret)
        return psgs


TOKEN = YOUR_HF_TOKEN

class BasicGenerator:
    def __init__(self, model_name_or_path, load = 0):
        logger.info(f"Loading model from {model_name_or_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_auth_token = TOKEN)
        #self.model_config = AutoConfig.from_pretrained(model_name_or_path,
        #            trust_remote_code = "falcon" in model_name_or_path, use_auth_token = TOKEN)
        #self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, device_map="auto", torch_dtype=torch.bfloat16,
        #            trust_remote_code = "falcon" in model_name_or_path, use_auth_token = TOKEN)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.bfloat16,
                    trust_remote_code = "falcon" in model_name_or_path, use_auth_token = TOKEN)
        
        #self.model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf", device_map="auto",
        #            trust_remote_code = "falcon" in model_name_or_path, use_auth_token = TOKEN)

        if load == 1:
            print("load models!!")
            from peft import LoraConfig, get_peft_model, TaskType
            from peft import PeftConfig, PeftModel
            lora_config = LoraConfig(
                                    r=8,
                                    lora_alpha=32,
                                    target_modules=["q_proj", "v_proj"],
                                    lora_dropout=0.05,
                                    inference_mode = False,
                                    bias="none",
                                    task_type=TaskType.CAUSAL_LM
                                    )
            lora_config = LoraConfig.from_pretrained("/home/user10/DRAGIN/matscinlp_dataset/save_model/llama_v2_matscinlp_prompt_ver_epoch_20")
            
            #self.model = get_peft_model(self.model, lora_config)
            self.model = PeftModel.from_pretrained(self.model, "/home/user10/DRAGIN/matscinlp_dataset/save_model/llama_v2_matscinlp_prompt_ver_epoch_20")
            #self.model.from_pretrained("/home/user10/DRAGIN/matscinlp_dataset/save_model/llama_v2_matscinlp")
            self.model.print_trainable_parameters()


        #self.model.cuda()
        self.model.eval()
        
        #self.model.cpu()
        
        #self.device = torch.device("cuda:1")
        #self.model.to(self.device)
        
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.model.generation_config.pad_token_id = self.tokenizer.pad_token_id
        self.tokenizer.padding_side = "left" 
        #self.model.to(device)

        self.space_token = self.tokenizer.tokenize(' ')[0]
        '''
        self.model_config = self.model.config
        if self.model_config.model_type == "llama":
            self.space_token = "▁"
        else:
            self.space_token = self.tokenizer.tokenize(' ')[0]
        '''
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token




    def generate(self, input_text, max_length, return_logprobs=False, batch_decode = False):
        #input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        if batch_decode:
            #input_ids = self.tokenizer(input_text, return_tensors="pt", padding = True)
            #if chat_template:
            #    input_ids = self.tokenizer.apply_chat_template(input_text, return_tensors="pt", padding = True)
            #else:
            input_ids = self.tokenizer(input_text, return_tensors="pt", padding = True)
            
            #print(self.tokenizer.decode(input_ids["input_ids"][0]))          
            #print(input_ids)
            
            attention_mask = input_ids["attention_mask"].to(self.device)
            input_ids = input_ids["input_ids"]
        else:
            input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
            #attention_mask = torch.ones_like(input_ids).cuda()
            attention_mask = torch.ones_like(input_ids).to(self.device)
            
        
        #input_ids = input_ids.to(device)
        input_ids = input_ids.to(self.device)
        
        #print(input_text)
        #print(len(input_text))
        
        #print(input_ids, attention_mask)
        #print(input_ids.shape)
        
        
        input_length = input_ids.shape[1]

        if return_logprobs:
            outputs = self.model.generate(
                input_ids = input_ids, 
                attention_mask = attention_mask,
                max_new_tokens = max_length, 
                return_dict_in_generate = True, 
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                num_beams=1,
                output_scores = True,
            )
            
            #print(outputs)
            
            generated_tokens = outputs.sequences[:, input_length:]
            
            #probabilities = torch.softmax(outputs.scores[-1], dim=-1)[0]
            
            #print(transition_scores)
            
            if batch_decode:
                text = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens = True)
                #text = [self.tokenizer.decode(t) for t in generated_tokens]
                tokens = []
                for sentence in generated_tokens:
                    tokens += [self.tokenizer.decode(t, skip_special_tokens = True) for t in sentence]
                '''
                logprobs = []
                for sentence in probabilities:
                    logprobs += [p.cpu().numpy() for p in sentence]
                '''
                logprobs = torch.softmax(outputs.scores[-1], dim=-1)#[0]
                logprobs = logprobs.cpu().numpy()
                #print(logprobs.shape)
                
            else:
                transition_scores = self.model.compute_transition_scores(
                    outputs.sequences, outputs.scores, normalize_logits=True
                )

                text = self.tokenizer.decode(generated_tokens[0], skip_special_tokens = True)
                tokens = [self.tokenizer.decode(t, skip_special_tokens = True) for t in generated_tokens[0]]
                logprobs = transition_scores[0]
                logprobs = [p.cpu().numpy() for p in logprobs]
            
            #text = self.tokenizer.decode(generated_tokens[0]) # text = "".join(tokens)

            

            #assert len(tokens) == len(logprobs)
            return text, tokens, logprobs
        
        else:
            outputs = self.model.generate(
                input_ids = input_ids, 
                max_new_tokens = max_length, 
                attention_mask = attention_mask,
            )
            #print(outputs.shape)
            
            generated_tokens = outputs[:, input_length:]
            
            #print(generated_tokens.shape)
            
            if batch_decode:
                text = self.tokenizer.batch_decode(generated_tokens)
                #text = [self.tokenizer.decode(t) for t in generated_tokens]
            else:
                text = self.tokenizer.decode(generated_tokens[0])
            #text = self.tokenizer.batch_decode(generated_tokens)
            
            return text, None, None
    
    def generate_attn(self, input_text, max_length, solver="max", use_entropy = False, use_logprob = False):
        input_ids = self.tokenizer.encode(input_text, return_tensors="pt")
        input_ids = input_ids.to(self.model.device)
        input_length = input_ids.shape[1]
        attention_mask = torch.ones_like(input_ids)

        outputs = self.model.generate(
            input_ids = input_ids, 
            attention_mask = attention_mask,
            max_new_tokens = max_length, 
            return_dict_in_generate = True, 
            output_scores = True,
        )
        generated_tokens = outputs.sequences[:, input_length:]
        tokens = self.tokenizer.convert_ids_to_tokens(generated_tokens[0])
        text = self.tokenizer.decode(generated_tokens[0])

        # merge tokens
        range_ = []
        for i, t in enumerate(tokens):
            if i == 0 or t.startswith(self.space_token) or generated_tokens[0][i] == 13 or tokens[i-1] == '</s>':
                range_.append([i, i])
            else:
                range_[-1][-1] += 1

        # attention
        atten = self.model(generated_tokens, output_attentions=True).attentions[-1][0]
        if solver == "max": 
            mean_atten, _ = torch.max(atten, dim=1)
            mean_atten = torch.mean(mean_atten, dim=0)
        elif solver == "avg":
            mean_atten = torch.sum(atten, dim=1)
            mean_atten = torch.mean(mean_atten, dim=0)
            for i in range(mean_atten.shape[0]):
                mean_atten[i] /= (mean_atten.shape[0] - i)
        elif solver == "last_token":
            mean_atten = torch.mean(atten[:, -1], dim=0)
        else:
            raise NotImplementedError
        if mean_atten.shape[0] > 1 and tokens[0] == '</s>':
            mean_atten = mean_atten / sum(mean_atten[1:]).item()
        # mean_atten = mean_atten[tl:tr]
            
        # regular tokens
        seqlist = []
        attns = []
        for r in range_:
            tokenseq = "".join(tokens[r[0]: r[1]+1]).replace(self.space_token, "")
            value = sum(mean_atten[r[0]: r[1]+1]).item()
            seqlist.append(tokenseq)
            attns.append(value)

        # -log prob
        if use_logprob:
            transition_scores = self.model.compute_transition_scores(
                outputs.sequences, outputs.scores, normalize_logits=True
            )
            logprobs = transition_scores[0]
            logprobs = [p.cpu().numpy() for p in logprobs]
            assert len(tokens) == len(logprobs)
            seqlogprobs = []
            for r in range_:
                logprobseq = sum(logprobs[r[0]:r[1]+1]) / (r[1] - r[0] + 1)
                seqlogprobs.append(logprobseq)
        else:
            seqlogprobs = None

        # entropy
        if use_entropy:
            tmp = []
            for v in outputs.scores:
                #print(v.shape)
                tmp.append(v.cpu())
                
            #print(tmp)
                
            softmax_probs = softmax(tmp, axis=-1)
            entropies = -np.sum(softmax_probs * np.log(softmax_probs + 1e-10), axis=-1)
            entropies = [v[0] for v in entropies]
            seqentropies = []
            for r in range_:
                entropyseq = sum(entropies[r[0]:r[1]+1]) / (r[1] - r[0] + 1)
                seqentropies.append(entropyseq) 
        else:
            seqentropies = None 

        return text, seqlist, attns, seqlogprobs, seqentropies
    
import spacy
class Counter:
    def __init__(self):
        self.retrieve = 0
        self.generate = 0
        self.hallucinated = 0
        self.token = 0
        self.sentence = 0

    def add_generate(self, text, tokenizer):
        self.generate += 1
        ids = tokenizer(text, return_tensors="pt")['input_ids'][0].tolist()
        self.token += len(ids)
        nlp = spacy.load("en_core_web_sm")
        sentences = [sent.text for sent in nlp(text).sents]
        self.sentence += len(sentences)

    def calc(self, other_counter):
        return {
            "retrieve_count": self.retrieve - other_counter.retrieve, 
            "generate_count": self.generate - other_counter.generate,
            "hallucinated_count": self.hallucinated - other_counter.hallucinated, 
            "token_count": self.token - other_counter.token, 
            "sentence_count": self.sentence - other_counter.sentence 
        }
    
from zebra.our_prompts_ckd import SHOT_TEMPLATES
class BasicRAG:
    qa_input_template = lambda self, ques: f'Question: {ques} \n\n Answer:'
    def __init__(self, args, templates=None):
        args = args.__dict__
        for k, v in args.items():
            setattr(self, k, v)
        print(args)

        #self.generator = BasicGenerator(self._name_or_path, args["load"])
        self.generator = BasicGenerator(self.model_name_or_path)
        
        self.retriever = DPR(
            retrieval_model_name_or_path = self.retrieval_model_name_or_path,
            embedding_path = self.embedding_path,
            passage_file = None
        )

        self.counter = Counter()
        self.templates = templates
        
        self.device = torch.device("cuda:0")
        self.generator.model.to(self.device)

        if templates != None:
            self.inst = templates[0]
            self.reply = templates[1]
            self.answer = templates[2]
            self.knowledge_inst = templates[3]

    def retrieve(self, query, topk=1, max_query_length=64):
        
        self.counter.retrieve += 1
        docs = self.retriever.retrieve(
            queries = [query],
            topk = topk
        )
         
        return docs[0]

    def get_top_sentence(self, text):
        sentences = [sent.text.strip() for sent in nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]
        return sentences[0] if len(sentences) > 0 else ""

    def get_last_sentence(self, text):
        sentences = [sent.text.strip() for sent in nlp(text).sents]
        sentences = [sent for sent in sentences if len(sent) > 0]
        return sentences[-1] if len(sentences) > 0 else "" 
    
    def inference(self, question, demo, case):
        # non-retrieval
        #assert self.query_formulation == "direct"
        prompt = "".join([d["case"]+"\n" for d in demo])
        prompt += case
        
        #print(prompt)
        
        text, _, _ = self.generator.generate(prompt, self.generate_max_length)
        if self.use_counter == True:
            self.counter.add_generate(text, self.generator.tokenizer)
        return text, None, None
    
    def tokenizer_handler(
            self,
            inputs
        ):
        inputs = inputs[:, :-1]
        return inputs
    
    def generate_text(
        self,
        prompt_list,
        max_new_tokens = 256):
        """
        Generate text using the model.

        Parameters:
        - model (AutoModelForCausalLM): Model.
        - tokenizer (AutoTokenizer): Tokenizer.
        - prompt (List[str]): List of conversation turns.
        - max_new_tokens (int): Maximum number of new tokens.
        - device (str): Device to run the model on (default is 'cuda' if available, otherwise 'cpu').

        Returns:
        - tuple: A tuple containing the model outputs and the generated text.
        """
        # Build the conversation turns in the format required by chat templates.
        
        input_list = []
        #attention_mask = []
        
        for i in range(len(prompt_list)):
            prompt = prompt_list[i]            
            messages = []
            for turn_id, turn in enumerate(prompt):
                if turn_id % 2 == 0:
                    messages.append({"role": "user", "content": turn})
                else:
                    messages.append({"role": "assistant", "content": turn})

            inputs = self.generator.tokenizer.apply_chat_template(messages, return_tensors="pt")#.to(device)
            inputs = self.tokenizer_handler(inputs)
        
            input_list.append(inputs[0])
        
        
        max_length = [len(s) for s in input_list]
        max_length = max(max_length)
        
        input_ids = torch.zeros((len(input_list), max_length)).to(torch.long).fill_(self.generator.tokenizer.pad_token_id)
        attention_mask = torch.zeros((len(input_list), max_length)).to(torch.long)
        
        for idx in range(len(input_list)):
            length = len(input_list[idx])
            #print(input_list[idx])
            
            input_ids[idx, -length:] = input_list[idx]
            attention_mask[idx, -length:] = 1
        
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        
        #attention_mask = torch.ones_like(inputs).to(device)
        
        outputs = self.generator.model.generate(
            input_ids = input_ids, 
            attention_mask = attention_mask,
            max_new_tokens = max_new_tokens, 
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            num_beams=1,
            return_dict_in_generate = True, 
            output_scores = True,
        )

        input_length = input_ids.shape[1]

        generated_tokens = outputs.sequences[:, input_length:]

        text = self.generator.tokenizer.batch_decode(generated_tokens, skip_special_tokens = True)
        #text = [self.tokenizer.decode(t) for t in generated_tokens]
        tokens = []
        for sentence in generated_tokens:
            tokens += [self.generator.tokenizer.decode(t, skip_special_tokens = True) for t in sentence]

        logprobs = torch.softmax(outputs.scores[-1], dim=-1)#[0]
        logprobs = logprobs.cpu().numpy()

        
        return text, tokens, logprobs
    
    def inference_batch(self, question, choices):
        # non-retrieval
        prompt_list = []
        for idx in range(len(question)):
            #prompt = "".join([d["case"]+"\n" for d in demo[i]])
            #prompt += case[i]
            q = question[idx]
            c = choices[idx]
            c = [f"* {c_['label']}: {c_['text']}".strip() for c_ in c]
            c = "\n".join(c)
            
            prompt = [self.inst, self.reply]
            final_shot = SHOT_TEMPLATES["mcq"].format(question=q, choices=c)
            assistant_reply = "Answer: "
            prompt += [final_shot, assistant_reply]
            
            prompt_list.append(prompt)
    
        #if log_prob:
        text, tokens, logprobs = self.generate_text(prompt_list, max_new_tokens = 1)
        #else:
        #    text, tokens, logprobs = self.generate_text(prompt_list, max_new_tokens = 1)
        
        if self.use_counter == True:
            for i in range(len(question)):
                self.counter.add_generate(text[i], self.generator.tokenizer)    
        return text, None, None, logprobs

class SingleRAG(BasicRAG):
    def __init__(self, args, templates = None):
        super().__init__(args, templates = None)
        self.device = torch.device("cuda:0")
        self.generator.model.to(self.device)
    def inference(self, question, demo, case):
        
        #print(question)
        #print("================= demo ===============")
        #print(demo)
        #print("================= case ===============")
        #print(case)
        #assert self.query_formulation == "direct"
        docs = self.retrieve(question, topk=self.retrieve_topk)
        '''
        docs = []
        for query in question:    
            #query = splited[i]
            d = self.retrieve(query, topk = self.retrieve_topk)
        '''
        #print("================= retrievaled documents ===============")
        #print(docs)
        
        # 对 topk 个 passage 生成 prompt
        prompt = "".join([d["case"]+"\n" for d in demo])
        #prompt += "Context:\n"
        prompt += "Below are the external knowledge references:\n"
        #print(docs)
        for i, doc in enumerate(docs):
            prompt += f"[{i+1}] {doc}\n"
        #prompt += "Answer in the same format as before.\n"
        prompt += "Please answer the question based on the external knowledge.\n"
        
        prompt += case
        
        
        print(prompt)
        text, _, _ = self.generator.generate(prompt, self.generate_max_length)
        if self.use_counter == True:
            self.counter.add_generate(text, self.generator.tokenizer)
        return text, docs, [question]
    
    def inference_batch2(self, question, demo, case, log_prob = False, batch_decode = True, query_expansion = False):
        #assert self.query_formulation == "direct"
        doc_list = []
        for query in question:    
            d = self.retrieve(query, topk = self.retrieve_topk)
            doc_list.append(d)
        print(doc_list)

        prompt_list = []
        for i in range(len(demo)):
            # 对 topk 个 passage 生成 prompt
            prompt = "".join([d["case"]+"\n" for d in demo[i]])
            #prompt += "Context:\n"
            prompt += "Below are the external knowledge references:\n"

            docs = doc_list[i]
            for j, doc in enumerate(docs):
                prompt += f"[{j+1}] {doc}\n"
            prompt += "Please answer the question based on the external knowledge.\n"
            if self.templates != None:
                prompt = prompt + self.inst + self.qa_input_template(case[i])
            else:
                prompt += case[i]
            #print(prompt)
            
            prompt_list.append(prompt)

        if log_prob:
            text, tokens, logprobs = self.generator.generate(prompt_list, self.generate_max_length, return_logprobs=True, batch_decode = True)
        else:
            text, tokens, logprobs = self.generator.generate(prompt_list, self.generate_max_length, batch_decode = True)
        
        if self.use_counter == True:
            for i in range(len(demo)):
                self.counter.add_generate(text[i], self.generator.tokenizer)
            
        return text, doc_list, question, logprobs, None

    
    def inference_batch(self, question, choices, inst, reply):
        #assert self.query_formulation == "direct"
        doc_list = []
        for query in question:    
            #d = self.retrieve(query, topk = self.retrieve_topk)
            d = self.retrieve(query, topk = 10)
            doc_list.append(d)

        prompt_list = []
        for idx in range(len(question)):
            # 对 topk 个 passage 生成 prompt
            prompt = ""
            #prompt += "Context:\n"
            prompt += "Below are the external knowledge references:\n"

            docs = doc_list[idx]
            for j, doc in enumerate(docs):
                prompt += f"[{j+1}] {doc}\n"
            prompt += "Please answer the question based on the external knowledge.\n"

            q = question[idx]
            c = choices[idx]
            c = [f"* {c_['label']}: {c_['text']}".strip() for c_ in c]
            c = "\n".join(c)
            
            #print(prompt)
            #prompt = [prompt + self.inst, self.reply]
            # ⭐⭐⭐ task 별 inst reply 수정
            #inst = "You are a helpful assistant for question answering."
            #reply = "You are given a question and up to 5 options (labeled A, B, C, D, and E). "
            prompt = [prompt + inst, reply]
            final_shot = SHOT_TEMPLATES["mcq"].format(question=q, choices=c)
            assistant_reply = "Answer: "
            prompt += [final_shot, assistant_reply]
            
            prompt_list.append(prompt)

        text, tokens, logprobs = self.generate_text(prompt_list, max_new_tokens = 1)
        #if self.use_counter == True:
        #    for i in range(len(question)):
        #        self.counter.add_generate(text[i], self.generator.tokenizer)
        for i in range(len(question)):
            self.counter.add_generate(text[i], self.generator.tokenizer)

            
        return text, doc_list, question, logprobs

    def inference_batch_speculative(self, question, choices, responses, inst, reply):
        #assert self.query_formulation == "direct"
        doc_list = []
        for query in question:    
            #d = self.retrieve(query, topk = self.retrieve_topk)
            d = self.retrieve(query, topk = 10)
            doc_list.append(d)

        prompt_list = []
        for idx in range(len(question)):
            # 对 topk 个 passage 生成 prompt
            prompt = ""
            prompt += "Below are the external knowledge references:\n"
            docs = doc_list[idx]
            for j, doc in enumerate(docs):
                prompt += f"[{j+1}] {doc}\n"

            prompt += "And below is the draft from external knowledge references: \n"
            r = responses[idx]
            prompt += r
            prompt += "Please answer the question based on the external knowledge and draft.\n"

            q = question[idx]
            c = choices[idx]
            c = [f"* {c_['label']}: {c_['text']}".strip() for c_ in c]
            c = "\n".join(c)
            
            prompt = [prompt + inst, reply]
            final_shot = SHOT_TEMPLATES["mcq"].format(question=q, choices=c)
            assistant_reply = "Answer: "
            prompt += [final_shot, assistant_reply]
            print(prompt)
            
            prompt_list.append(prompt)

        text, tokens, logprobs = self.generate_text(prompt_list, max_new_tokens = 1)
        for i in range(len(question)):
            self.counter.add_generate(text[i], self.generator.tokenizer)
            
        return text, doc_list, question, logprobs

    def inference_batch_speculative2(self, question, choices, responses, inst, reply):
        #assert self.query_formulation == "direct"
        #doc_list = []
        #for query in question:    
        #    #d = self.retrieve(query, topk = self.retrieve_topk)
        #    d = self.retrieve(query, topk = 10)
        #    doc_list.append(d)

        prompt_list = []
        for idx in range(len(question)):
            # 对 topk 个 passage 生成 prompt
            prompt = ""
            prompt += "## Drafted Context from Retrieved External Knowledge:\n"

            #docs = doc_list[idx]
            #for j, doc in enumerate(docs):
            #    prompt += f"[{j+1}] {doc}\n"

            #r = responses[idx][0]
            r = responses[idx]
            prompt += r
            prompt += " Based on the above draft, answer the following question by selecting the most appropriate choice.\n\n"

            q = question[idx]
            c = choices[idx]
            c = [f"* {c_['label']}: {c_['text']}".strip() for c_ in c]
            c = "\n".join(c)
            
            prompt = [prompt + inst, reply]
            final_shot = SHOT_TEMPLATES["mcq"].format(question=q, choices=c)
            assistant_reply = "Answer: "
            prompt += [final_shot, assistant_reply]
            #print(prompt)
            
            prompt_list.append(prompt)

        text, tokens, logprobs = self.generate_text(prompt_list, max_new_tokens = 1)
        for i in range(len(question)):
            self.counter.add_generate(text[i], self.generator.tokenizer)
        
        _ = 0
            
        return text, _, question, logprobs

    def retrieve_and_save(self, question, choices):
        #assert self.query_formulation == "direct"
        doc_list = []
        for query in question:    
            #d = self.retrieve(query, topk = self.retrieve_topk)
            d = self.retrieve(query, topk = 5)
            doc_list.append(d)

        prompt_list = []
        for idx in range(len(question)):
            # 对 topk 个 passage 生成 prompt

            docs = doc_list[idx]
            for j, doc in enumerate(docs):
                prompt += f"[{j+1}] {doc}\n"

            q = question[idx]
            c = choices[idx]
            c = [f"* {c_['label']}: {c_['text']}".strip() for c_ in c]
            c = "\n".join(c)
            
            docs_list.append(prompt)

        return docs_list

import os
import json
import argparse
from tqdm import tqdm
from copy import copy
import logging
#from data import StrategyQA, WikiMultiHopQA, HotpotQA, IIRC, MatSciNLP
#from zebra_data import CSQA, CSQA2, PIQA

import torch
import random
import pickle

logging.basicConfig(level=logging.INFO) 
logger = logging.getLogger(__name__)

random.seed(42)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config_path", type=str, required=True)
    parser.add_argument("--query_expansion", action='store_true')
    parser.add_argument("--knowledge_generation", action='store_true')
    parser.add_argument("--retrieve_topk", type = int, default = 3)
    parser.add_argument("--output_dir", type=str, default="/home/user4/Speculative_RAG/data/coconut_documents/output")
    parser.add_argument("--data_path", type=str, default="/home/user4/RALM_CSQA/data/zebra/arc-challenge")
    parser.add_argument("--dataset", type=str, default="arc-challenge")
    parser.add_argument("--method", type=str)
    parser.set_defaults(query_expansion=False)
    parser.set_defaults(knowledge_generation=False)       
    
    #parser.add_argument("--load", type = int, default = 0)
    args = parser.parse_args()
    #print(args.query_expansion, args.knowledge_generation)
    config_path = args.config_path
    args_list = vars(args)
    with open(config_path, "r") as f:
        temp = json.load(f)
        for k,v in temp.items():
            if isinstance(v, bool):
                v = str(v).lower()  # argparse는 문자열을 사용하므로 bool을 문자열로 변환
            
            if k in args_list.keys():
                continue
            
            parser.add_argument(f'--{k}', type=type(v), default=v)
            

    args = parser.parse_args()
    
    #args = argparse.Namespace(**args)
    args.config_path = config_path
    if "shuffle" not in args:
        args.shuffle = False 
    if "use_counter" not in args:
        args.use_counter = True
    if "zeroshot" not in args: 
        args.zeroshot = True
    if "augmentation" not in args:
        args.augmentation = False
    args.stepback = 0
    #print(args.stepback, args.load, args.query_expansion, args.knowledge_generation)

    return args

def get_model_answer(
    log_prob,
    tokenizer,
    labels,
    return_scores=False,
):
    """
    Get the answer from the model.

    Parameters:
    - tokenizer (AutoTokenizer): Tokenizer.
    - labels (Optional[List[str]]): Labels, default is ["A", "B", "C", "D", "E"].
    - return_scores (Optional[bool]): Return scores.

    Returns:
    - str: Answer (one of the labels).
    - Optional[torch.Tensor]: Scores (if return_scores is True).
    """
    # Get the probabilities of the first token.
    #probabilities = torch.softmax(log_prob, dim=-1)[0]

    probabilities = log_prob
    
    
    #print(probabilities)

    # Check that the labels are in the tokenizer's vocabulary.
    labels = [label for label in labels if len(tokenizer.tokenize(label)) == 1]

    # Get the label IDs.
    label_ids = [
        tokenizer.encode(label, add_special_tokens=False)[0] for label in labels
    ]

    # Get the probability of each label (A, B, C, D, E) and its variants.
    answer = [probabilities[label_id].item() for label_id in label_ids]

    # Get the label with the highest probability.
    answer = labels[answer.index(max(answer))]
    answer = answer.lstrip()
    
    #print(answer)

    if return_scores:
        return answer, probabilities
    else:
        return answer

def main():
    print(0)

    args = get_args()
    logger.info(f"{args}")

    # output dir
    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)
    dir_name = os.listdir(args.output_dir)
    for i in range(10000):
        if (args.method + "_%d_stepback_%d"%(i, args.stepback)) not in dir_name:
            args.output_dir = os.path.join(args.output_dir, args.method + "_%d_stepback_%d"%(i, args.stepback))
            os.makedirs(args.output_dir)
            break
    logger.info(f"output dir: {args.output_dir}")
    # save config
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(args.__dict__, f, indent=4)
    # create output file
    output_file = open(os.path.join(args.output_dir, "output.txt"), "w")
    
    output_file2 = open(os.path.join(args.output_dir, "output_explanation.txt"), "w")


    if args.method in ["ours", "zebra"]:
        from zebra_data import CSQA, CSQA2, PIQA, HellaSWAG
    else:
        from data import CSQA, CSQA2, PIQA

    # load data
    from zebra_data import OOD_Dataset, ID_Dataset

    # load data
    if args.dataset in ["csqa", "csqa2", "piqa", "arc", "obqa", "qasc", "wg", "arc-challenge", "arc-easy"]:
        data = ID_Dataset(args, args.data_path)    
    elif args.dataset in ["siqa","hellaswag","riddlesense","com2sense", "numersense", "quartz"]:
        data = OOD_Dataset(args, args.data_path)
    else:
        raise NotImplementedError


    if args.method == "non-retrieval":
        model = BasicRAG(args)
    elif args.method == "single-retrieval":
        model = SingleRAG(args)
    

    labels = data.labels
    # Generate alternative labels with whitespaces in front.
    labels.extend([f" {label}" for label in labels])

    data.format(fewshot=args.fewshot)
    data = data.dataset
    
    print(len(data))
    
    if args.shuffle:
        data = data.shuffle()
    if args.sample != -1:
        samples = min(len(data), args.sample)
        data = data.select(range(samples))

    
    batch_size = 8
    device = torch.device("cuda")
    logger.info("start inference")
    ans_list = []
    gt_ans_list = []
    for i in tqdm(range(len(data) // batch_size), ncols = 100):
        #last_counter = copy(model.counter)
        batch = data[batch_size * i : batch_size * (i+1)]
        
        if args.zeroshot:
            
            #pred, docs, quries, log_prob, explanations = model.inference_batch(batch["question"], batch["choice"], query_expansion= args.query_expansion, knowledge_generation = args.knowledge_generation)
            pred, docs, quries, log_prob, explanations = model.inference_batch(batch["question"], batch["choice"])
            
            #pred, docs, quries, log_prob, explanations = model.inference_batch_cluster(batch["question"], batch["choice"], query_expansion= True, knowledge_generation = True)
            
            #print(pred)
            #pred, docs, quries, log_prob, explanations = model.inference_batch2(batch["question"], batch["choice"], summarization = args.summarization)
            #pred, docs, quries, log_prob, explanations = model.inference_batch(batch["question"], batch["choice"])
            #pred, docs, quries, log_prob, explanations = model.inference_batch(batch["question"], batch["choice"], query_expansion = True)

            pred_ans = [get_model_answer(prob, model.generator.tokenizer, labels) for prob in log_prob]
            ans_list += pred_ans
            gt_ans_list += batch["answer"]
            
            #print(labels)
        else:
            pred, docs, quries, _ = model.inference_batch(batch["question"], batch["demo"], batch["case"], batch_decode = True)
        
        
        for j in range(batch_size):
            if args.method == "non-retrieval":
                ret = {
                    'question' : batch["question"][j],
                    "prediction": pred[j].strip(),
                    "answer" : batch["answer"][j],
                    "qid": batch["qid"][j],
                }
            else:
                ret = {
                    "prediction": str(pred[j]),
                    "answer" : batch["answer"][j],
                    "quries" : str(quries[j]),
                    "explain" : str(explanations[j]),
                    "docs" : str(docs[j]),
                    'question' : batch["question"][j],
                    #'choice' : batch["choice"][j]
                    #"qid": batch["qid"][j],
                }
                
            ret2 = {
                "explain" : str(explanations[j]),
                "docs" : str(docs[j])
            }
                
            output_file.write(json.dumps(ret)+"\n")
            output_file2.write(json.dumps(ret2) + "\n")


    if args.zeroshot:

        
        result_file = open(os.path.join(args.output_dir, "result.txt"), "w")

        score = 0
        for p_a, g_a in zip(ans_list, gt_ans_list):
            if p_a == g_a:
                score += 1
        
        score /= len(ans_list)
        
        print(f"\nAcc : {score}\n")
        
        result_file.write(f"\nAcc : {score}\n")
        result_file.close()

        with open("./total_results.txt","a") as f:
            f.write(os.path.join(args.output_dir, args.method + f"_{i}_{args.query_expansion}_{args.knowledge_generation}_{args.retrieve_topk}") + f" | Acc : {score}\n")

    output_file.close()
    output_file2.close()


if __name__ == "__main__":
    main()
