import json
import pickle as pkl
import numpy as np
import os
import argparse
import matplotlib.pyplot as plt

from transformers import pipeline, AutoTokenizer, AutoModelForMaskedLM, AutoModelForCausalLM, XGLMTokenizer, T5ForConditionalGeneration, MT5ForConditionalGeneration
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm
import torch

def contain(answer_tokens, orig_text_tokens):
    ans_len = len(answer_tokens)

    for i in range(len(orig_text_tokens)-ans_len+1):
        if orig_text_tokens[i: i+ans_len] == answer_tokens:
            return True

    return False

def nooverlap(a, b):
    return len(a) + len(b) == len(list(set(a + b)))

def predict_mask(answer_cand, prompt, mname):
    answer_pred_probs = dict()
    
    for answer in answer_cand:
        answer_cand_probs = []
    
        if "t5" not in mname and "xglm" not in mname:
            answer_tokens = tokenizer(answer)["input_ids"][1:-1]
    
            if "xlm-roberta" in mname and answer_tokens[0] == 6 and lang == "zh":
                answer_tokens = answer_tokens[1:]
    
            new_mask = ["<mask>"] * len(answer_tokens)
    
            if lang == "zh":
                new_mask = "".join(new_mask)
            else:
                new_mask = " ".join(new_mask)
    
            prompt_new = prompt.replace('<mask>', new_mask)
            prompt_new = prompt_new.replace('<mask>', tokenizer.mask_token)
    
            for j, w_idx in enumerate(answer_tokens):
                model_inputs = tokenizer(prompt_new, return_tensors='pt')
                model_outputs = model(**model_inputs)
                input_ids = model_inputs["input_ids"][0]
                outputs = model_outputs["logits"]
                masked_index = torch.nonzero(input_ids == tokenizer.mask_token_id, as_tuple=False)
                    
                logits = outputs[0, masked_index[0].item(), :]
                probs = logits.softmax(dim=-1).detach().numpy()
                answer_cand_probs.append(-np.log(probs[w_idx]))
    
                pos = prompt_new.find(tokenizer.mask_token)
                prompt_new = prompt_new[:pos] + tokenizer.convert_ids_to_tokens(w_idx) + prompt_new[pos+len(tokenizer.mask_token):]
    
            answer_pred_probs[answer] = np.mean(answer_cand_probs)
    
        elif "xglm" in mname:
            prompt_new = prompt.replace("<mask>", answer)
                 
            model_input = tokenizer(prompt_new, return_tensors='pt')
            output = model(**model_input)
                
            if lang == 'zh':
                logits = output['logits'][0, :-1] 
                token_ids = model_input['input_ids'][0, 1:]
            else:
                logits = output['logits'][0, :-2] 
                token_ids = model_input['input_ids'][0, 1:-1]
    
            answer_pred_probs[answer] = float(torch.nn.CrossEntropyLoss(reduction='mean')(logits, token_ids))
        else:
            input_ids = tokenizer(prompt.replace('<mask>', '<extra_id_0>'), return_tensors='pt').input_ids
            labels = tokenizer('<extra_id_0> ' + answer + ' <extra_id_1>', return_tensors='pt').input_ids
            target_ids = labels[0][1:-2]
    
            outputs = model(input_ids=input_ids, labels=labels).logits
            masked_index = torch.tensor(list(range(outputs.size()[1]))[1:-2])
            
            for idx, t_idx in zip(masked_index, target_ids):
                logits = outputs[0, idx.item(), :]
                probs = logits.softmax(dim=-1).detach().numpy()
                answer_cand_probs.append(-np.log(probs[t_idx]))
    
            answer_pred_probs[answer] = np.mean(answer_cand_probs)
            
    return answer_pred_probs
    
parser = argparse.ArgumentParser()

parser.add_argument(
    '-lang',
    dest='lang',
    default='en',
    help='language',
    type=str,
)

parser.add_argument(
    '-mname',
    dest='mname',
    default='bert-base-multilingual-cased',
    help='model name',
    type=str,
)

parser.add_argument(
    '-country_or_not',
    default='yes',
    help='contain country names or not',
    type=str,
)


args = parser.parse_args()

lang = args.lang
w_country = args.country_or_not
fname = "gd_prompt_" + lang + ".tsv"
aug_fname = "gd_prompt_" + lang + "_aug.tsv"
mname = args.mname

# mname = "xlm-mlm-100-1280"
# mname = "xlm-roberta-base"
# mname = "xlm-roberta-large"
# mname = "google/mt5-small"
# mname = "google/mt5-base"
# mname = "google/mt5-large"
# mname = "facebook/xglm-564M"
# mname = "facebook/xglm-1.7B"
# mname = "facebook/xglm-2.9B"
# mname = "facebook/xglm-4.5B"

if "xglm" in mname:
    model = AutoModelForCausalLM.from_pretrained(mname)
elif "google/mt5" in mname:
    model = MT5ForConditionalGeneration.from_pretrained(mname)
elif "t5" in mname:
    model = T5ForConditionalGeneration.from_pretrained(mname)
else:
    model = AutoModelForMaskedLM.from_pretrained(mname)

if "xglm" in mname:
    tokenizer = XGLMTokenizer.from_pretrained(mname)
else:
    tokenizer = AutoTokenizer.from_pretrained(mname)

data = []

with open(fname) as f:
    ans_cand = []

    for line in f:
        d = line.strip().split('\t')

        if d[0] == "Prompt":
            continue

        if len(d) <= 1:
            continue
        else:
            if len(d) == 5:
                d += ["", ""]
            d[-4] = d[-4].split(', ')
            d[-5] = d[-5].split(', ')
            
            data.append(d)

aug_data = []
with open(aug_fname) as f:
    idx = 0
    for line in f:
        d = line.strip().split('\t')
        d.append(data[int(idx/20)*5+idx%5][-5])
        d.append(data[int(idx/20)*5+idx%5][-4])
        aug_data.append(d)

        idx += 1

data += aug_data

s_corr = []
s_tot = []
all_gold_ans = []
answer_pred_orig_probs = dict()

country_corr = [0] * 5
country_tot = [0] * 5
country_tot_ans_cand = [0] * 5
country_corr_gpt = [0] * 5
country_ir_gpt = [0] * 5

country_list = ["en", "zh", "hi", "fa", "sw"]

tot_0 = 0.0
tot_1 = 0.0

concept = [0.0] * 25
concept_corr = [0.0] * 25
concept_tot = [0.0] * 25

for i, d in enumerate(data):
    print(i, '/', len(data))
    flag = False
    if i < 125:
        concept_id = int(i/5)
        if i % 5 == 0:
            flag = True
    else:
        concept_id = int((i-125)/20)
        if (i-125) % 20 == 0:
            flag = True

    prompt = d[0]

    if len(d) > 3:
        answer_cand = d[-4]
        gold_ans_list = d[-5]
    else:
        answer_cand = d[-1]
        gold_ans_list = d[-2]
    
    all_gold_ans.append(gold_ans_list)

    if len(answer_cand) == len(gold_ans_list):
        s_corr.append(0)
        s_tot.append(0)
        continue
    
    tot_0 += len(gold_ans_list)
    tot_1 += len(answer_cand)

    if i % 5 == 4:
        print("---")

    if flag:
        prompt_wo_country = prompt
        if lang == "en":
            prompt_wo_country = prompt_wo_country.lower()
            prompt_wo_country = prompt_wo_country.replace("the united states", "united states").replace("in united states, ", "").replace("of united states", "").replace("american", "")
            prompt_wo_country = prompt_wo_country.replace("in united states", "").replace("  ", " ")
            if prompt_wo_country[0] == ' ':
                prompt_wo_country = prompt_wo_country[1:]
        elif lang == "zh":
            prompt_wo_country = prompt_wo_country.replace("美国的", "美国").replace("美国人的", "").replace("在美国，", "").replace("在美国", "").replace("美国", "")
        elif lang == "fa":
            prompt_wo_country = prompt_wo_country.replace("در ایالت متحده آمریکا, ", "").replace("در ایالت متحده آمریکا", "").replace("در ایالت متحده آمریکا", "").replace("ایالت متحده آمریکا", "").replace("از ایالت متحده آمریکا", "")
        elif lang == "hi":
            prompt_wo_country = prompt_wo_country.replace("अमेरिका में", "").replace("अमेरिकी", "").replace("अमेरिका ", "").replace("अमेरिका के", "").replace("अमेरिका का", "")
        else:
            prompt_wo_country = prompt_wo_country.replace("Marekani", "").replace("Wamarekani", "Watu").replace("Kimarekani", "").replace("wamarekani", "watu")

        answer_pred_orig_probs = predict_mask(answer_cand, prompt_wo_country, mname)
    
    if w_country == 'yes':
        answer_pred_probs = predict_mask(answer_cand, prompt, mname)
        for k in answer_pred_orig_probs:
            answer_pred_probs[k] -= answer_pred_orig_probs[k]
    else:
        answer_pred_probs = answer_pred_orig_probs
    
    minv = 10000
    mink = ""
    sorted_probs = []
    for k in answer_pred_probs:
        sorted_probs.append(answer_pred_probs[k])

    sorted_probs = np.sort(np.array(sorted_probs))
    corr = 0
    tot = len(gold_ans_list)
    for gold_ans in gold_ans_list:
        if answer_pred_probs[gold_ans] in sorted_probs[:tot]:
            corr += 1
    
    pred = []
    for k in answer_pred_probs:
        if answer_pred_probs[k] in sorted_probs[:tot]:
            pred.append(k)

    s_corr.append(corr)
    s_tot.append(tot)
    country_corr[i%5] += corr
    country_tot[i%5] += tot
    country_tot_ans_cand[i%5] += len(answer_cand)
    concept_corr[concept_id] += corr
    concept_tot[concept_id] += tot
        
    if len(d) > 3:
        if d[-1] == 'T':
            country_corr_gpt[i%5] += len(d[-5])
        elif d[-1] == 'I':
            country_ir_gpt[i%5] += len(d[-5])
        elif d[-1] != 'F':
            if '/' in d[-1]:
                corr_num = int(d[-1].split('/')[0])
                country_corr_gpt[i%5] += corr_num

print(mname+'_'+lang+':', [country_corr[0]/145.0, country_corr[1]/140.0, country_corr[2]/165.0, country_corr[3]/145.0, country_corr[4]/160.0])
