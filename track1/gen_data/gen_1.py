import random
import json
from ltp import LTP
from tqdm import tqdm

ltp = LTP()

if __name__ == '__main__':

    n = 20000
    cls = '[CLS]'
    sep = '[SEP]'
    new_lines = []
    with open('new_train.txt', 'r',encoding='utf-8') as f:
        lines = f.readlines()
        for j in tqdm(range(50000)):
            i = n + j
            line = lines[i].strip('\n')
            result =ltp.pipeline([line], tasks = ["cws","pos"])
            cws = result.cws[0]
            pos = result.pos[0]
            res = list(zip(cws,pos))
            sum_char = 0
            for (c,p) in res:
                if p == 'n' or p == 'v':
                    sum_char += 1
            if sum_char != 0:
                i_char = 0
                tar_char = random.randint(1,sum_char)
                new_line = {}
                label = []
                source = cls
                target = cls + line + sep
                label.append('correct')
                for (c,p) in res:
                    if p == 'n' or p == 'v':
                        i_char += 1
                        if i_char == tar_char:
                            c_len = len(c)
                            del_index = random.randint(0,c_len-1)
                            for j in range(c_len):
                                if j == del_index:
                                    label[-1] = 'char_append'
                                else:
                                    source += c[j]
                                    label.append('correct')
                        else:
                            for word in c:
                                source += word
                                label.append('correct')
                    else:
                        for word in c:
                                source += word
                                label.append('correct')
                source += sep
                label.append('correct')
                new_line['source'] = source
                new_line['target'] = target
                new_line['label'] = label
                new_lines.append(new_line)

    with open("gen_1_label.json",'w',encoding='utf-8') as f:
        json.dump(new_lines,f,ensure_ascii=False,indent=1)