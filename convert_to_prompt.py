import json
import pickle as pkl
import numpy as np
import os
import argparse
import csv

parser = argparse.ArgumentParser(description='train')

parser.add_argument(
    '-lang',
    dest='lang',
    default='en',
    help='language',
    type=str,
)

args = parser.parse_args()

lang = args.lang
fname = "gd_prompt_" + lang + ".tsv"
aug_fname = lang + "_aug.tsv"

data = []
answers = []
with open(fname) as f:
    s = 0
    ans = []
    for line in f:
        d = line.strip().split('\t')

        if "prompt" in d[0].lower():
            continue

        if len(d) <= 1:
            continue
        else:
            s += 1
            ans += d[2].split(', ')
            if s == 5:
                s = 0
                answers.append(ans)
                ans = []
                print(answers[-1])

# country_name = ["अमेरिका", "चीन", "इंडिया", "ईरान", "केन्या"]
# country_name = ["آمریکا", "چین", "هند", "ایران", "کنیا"]
# country_name = ["the United States", "China", "India", "Iran", "Kenya"]
# country_name_1 = ["American", "Chinese", "Indian", "Iranian", "Kenyan"]
country_name = ["Marekani", "China", "India", "Uajemi", "Kenya"]
country_name_1 = ["Kimarekani", "Kichina", "Kihindi", "Kiajemi", "Kikenya"]
country_name_2 = ["Wamarekani", "Wachina", "Wahindi", "Waajemi", "Wakenya"]
# country_name = ["美国", "中国", "印度", "伊朗", "肯尼亚"]

data = []
aug_prompts = []
with open(aug_fname) as f:
    s = 0
    for line in f:
        '''if s == 0:
            s += 1
            continue'''
        
        ans = answers[int((s-0)/4)]
        # print(ans)
        # print(int((s-1)/4), ans)
        d = line.strip().split('\t')
        # prompt = d[-1]
        t = False
        prompt = d[-1]
        # print(prompt)
        # print("---")
        for a in ans:
            if a in prompt:
                t = True
                # print("kk", prompt)
                prompt = prompt.replace(a, "<mask>")

                if "N<mask>" in prompt:
                    prompt = prompt.replace("N<mask>", "Nchini")
                # if "<mask>" not in prompt:
                #     print(prompt)
                # print("kk", prompt)
                break
        
        # print(prompt)
        if not t:
            print("kkk", prompt)

        j = 0
        for country in country_name:
            '''if "China" in prompt:
                aug_prompts.append([prompt.replace("China", country)])
            elif "Chinese" in prompt:
                aug_prompts.append([prompt.replace("Chinese", country_name_1[j])])'''

            if "China" in prompt:
                aug_prompts.append([prompt.replace("China", country)])
            elif "Kichina" in prompt:
                aug_prompts.append([prompt.replace("Kichina", country_name_1[j])])
            elif "Wachina" in prompt or "wachina" in prompt:
                aug_prompts.append([prompt.replace("wachina", "Wachina").replace("Wachina", country_name_2[j])])
            
            # aug_prompts.append([prompt.replace("中国", country)])
            # aug_prompts.append([prompt.replace("चीन", country)])
            # aug_prompts.append([prompt.replace("چین", country)])

            j += 1

        s += 1
    print(s)
    # print(aug_prompts)

with open("gd_prompt_"+aug_fname, 'w') as f1:
    tsv_w = csv.writer(f1, delimiter='\t')
    tsv_w.writerows(aug_prompts)
