from transformers import FSMTForConditionalGeneration, FSMTTokenizer
from googletrans import Translator
import csv
from tqdm import tqdm

fname = "zh-aug.tsv"

data = []
with open(fname) as f:
    for line in f:
        d = line.strip().split('\t')
        print(d)
        data.append(d[0])

translator = Translator()
lang = 'sw'

fname_output = "gd_prompt_sw_aug_new.tsv"

with open(fname_output, 'w') as f:
    tsv_w = csv.writer(f, delimiter='\t')
    tsv_outputs = []
    for i in tqdm(range(len(data))):
        sent = data[i]
        if len(sent) == 0:
            tsv_outputs.append([sent])
        else:
            result = translator.translate(sent, src='en', dest=lang)
            print(sent, result.text)
            tsv_outputs.append([sent, result.text])
        
    tsv_w.writerows(tsv_outputs)
