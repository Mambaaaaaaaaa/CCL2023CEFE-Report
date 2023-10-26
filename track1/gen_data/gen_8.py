import random
import json
from ltp import LTP
from tqdm import tqdm


ltp = LTP()



if __name__ == '__main__':
    n = 750000
    cls = '[CLS]'
    sep = '[SEP]'
    new_lines = []
    with open('../source_data/filter_train.txt', 'r',encoding='utf-8') as f:
        lines = f.readlines()
        for j in tqdm(range(50000)):
            i = n + j
            line = lines[i].strip('\n')
            result =ltp.pipeline([line], tasks = ["cws","dep","pos"])
            cws = result.cws[0]
            label = result.dep[0]['label']
            pos = result.pos[0]
            res = list(zip(cws,label,pos))
            sum_other = 0
            for (c,l,p) in res:
                if l != 'VOB' and l != 'IOB' and l != 'FOB' and l != 'SBV' and p != 'v' and l != 'WP':
                    sum_other += 1
            if sum_other != 0:
                i_other = 0
                tar_other = random.randint(1,sum_other)
                new_line = {}
                label = []
                source = cls
                target = cls + line + sep
                label.append('correct')
                for (c,l,p) in res:
                    if l != 'VOB' and l != 'IOB' and l != 'FOB' and l != 'SBV' and p != 'v' and l != 'WP':
                        i_other += 1
                        if i_other == tar_other:
                            label[-1] = 'miss_other'
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
 
    with open("gen_8_label.json",'w',encoding='utf-8') as f:
        json.dump(new_lines,f,ensure_ascii=False,indent=1)

