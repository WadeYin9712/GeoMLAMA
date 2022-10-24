from transformers import FSMTForConditionalGeneration, FSMTTokenizer
from googletrans import Translator
import csv
from tqdm import tqdm

fname = "gd_prompt_en.tsv"

data = []
with open(fname) as f:
    for line in f:
        d = line.strip().split('\t')
        if d[0] == "Prompt":
            continue
        if len(d) <= 1:
            continue
        else:
            gold_ans = d[-5].split(", ")[0]
            data.append(d[0].replace("<mask>", gold_ans))

mname = "facebook/wmt19-de-en"
tokenizer_de2en = FSMTTokenizer.from_pretrained(mname)
model_de2en = FSMTForConditionalGeneration.from_pretrained(mname)

mname = "facebook/wmt19-en-de"
tokenizer_en2de = FSMTTokenizer.from_pretrained(mname)
model_en2de = FSMTForConditionalGeneration.from_pretrained(mname)

translator = Translator()
langs = ['sw']
en = []

for t in tqdm(range(len(data))):
    input = data[t]
    input_ids = tokenizer_en2de.encode(input, return_tensors="pt")
    outputs = model_en2de.generate(input_ids, num_beams=4, num_return_sequences=4)

    de = []
    for i in range(len(outputs)):
        de.append(tokenizer_en2de.decode(outputs[i], skip_special_tokens=True))

    para = []
    for i in range(len(de)):
        input_ids = tokenizer_de2en.encode(de[i], return_tensors="pt")
        outputs = model_de2en.generate(input_ids, num_beams=5, num_return_sequences=5)
        for j in range(len(outputs)):
            para.append(tokenizer_de2en.decode(outputs[j], skip_special_tokens=True))
    
    para = list(set(para))[:4]
    en += [[input, p] for p in para]

    if t % 4 == 3:
        en.append(["", ""])
        en.append(["", ""])

for lang in langs:
    if lang == 'zh-cn':
        fname_output = "gd_prompt_zh_aug.tsv"
    else:    
        fname_output = "gd_prompt_" + lang + "_aug.tsv"

    with open(fname_output, 'w') as f:
        tsv_w = csv.writer(f, delimiter='\t')
        tsv_outputs = []
        for i in tqdm(range(len(en))):
            sent = en[i]
            if len(sent[0]) == 0:
                tsv_outputs.append(sent)
                continue

            if lang != 'en':
                result = translator.translate(sent[1], src='en', dest=lang)
                print(sent[1], result.text)
                tsv_outputs.append([sent[0], result.text])
            else:
                tsv_outputs.append(sent)
        
        tsv_w.writerows(tsv_outputs)    
