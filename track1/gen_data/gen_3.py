import random
import json

from tqdm import tqdm

punc_list = ['！','？','｡','＂','＃','＄','％','＆','＇','（','）','＊','＋','，','－','／','：','；','＜','＝','＞','＠','［','＼','］','＾','＿','｀','｛','｜','｝','～','｟','｠','｢','｣','､','、','〃','》','「','」','『','』','【','】','〔','〕','〖','〗','〘','〙','〚','〛','〜','〝','〞','〟','〰','‘','‛','“','”','„','‟']

cls = '[CLS]'
sep = '[SEP]'
n = 200001
p = 0
new_lines = []

with open('../source_data/filter_train.txt', 'r',encoding='utf-8') as f:
    lines = f.readlines()
    for j in tqdm(range(200000)):
        i = n + j
        line = lines[i].strip('\n')
        sum_char = 0
        for word in line:
            if word in punc_list:
                sum_char += 1
        if sum_char > 0:
            tar_char = random.randint(1,sum_char)

            new_line = {}
            label = []
            source = cls
            target = cls + line + sep
            label.append('correct')

            i_char = 0
            for word in line:
                if word in punc_list:
                    i_char += 1
                    if i_char == tar_char:
                        label[-1] = 'char_punc_append'
                    else:
                        source += word
                        label.append('correct')
                else:
                    source += word
                    label.append('correct')
                
            source += sep
            label.append('correct')
            new_line['source'] = source
            new_line['target'] = target
            new_line['label'] = label
            assert (len(source)-len(label)) == 8
            new_lines.append(new_line)
            p += 1

with open("gen_3_label.json",'w',encoding='utf-8') as f:
    json.dump(new_lines,f,ensure_ascii=False,indent=1)

print(p)