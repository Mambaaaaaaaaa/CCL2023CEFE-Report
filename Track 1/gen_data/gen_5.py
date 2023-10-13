import random
import json
from ltp import LTP
from tqdm import tqdm

ltp = LTP()



if __name__ == '__main__':

    n = 600000
    cls = '[CLS]'
    sep = '[SEP]'
    new_lines = []
    with open('../source_data/filter_train.txt', 'r',encoding='utf-8') as f:
        lines = f.readlines()
        for j in tqdm(range(50000)):
            i = n + j
            line = lines[i].strip('\n')
            result =ltp.pipeline([line], tasks = ["cws","dep"])
            cws = result.cws[0]
            label = result.dep[0]['label']
            res = list(zip(cws,label))
            sum_sub = 0
            for (c,l) in res:
                if l == 'SBV':
                    sum_sub += 1
            if sum_sub != 0:
                i_sub = 0
                tar_sub = random.randint(1,sum_sub)
                new_line = {}
                label = []
                source = cls
                target = cls + line + sep
                label.append('correct')
                for (c,l) in res:
                    if l == 'SBV':
                        i_sub += 1
                        if i_sub == tar_sub:
                            label[-1] = 'miss_sub'
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

    with open("gen_5_label.json",'w',encoding='utf-8') as f:
        json.dump(new_lines,f,ensure_ascii=False,indent=1)


