import random
import json
from ltp import LTP

ltp = LTP()



if __name__ == '__main__':
    n = 0
    i = 3000
    cls = '[CLS]'
    sep = '[SEP]'
    new_lines = []
    with open('../source_data/new_train.txt', 'r',encoding='utf-8') as f:
        lines = f.readlines()
        while n != 1000:
            i += 1
            if i % 10 == 2:
                line = lines[i].strip('\n')
                result =ltp.pipeline([line], tasks = ["cws","dep","pos"])
                cws = result.cws[0]
                pos = result.pos[0]
                head = result.dep[0]['head']
                label = result.dep[0]['label']
                sum_sub = 0
                for j in range(len(cws)):
                    if pos[j] == 'v' and (label[j] == 'HED' or label[j] == 'COO'):
                        temp_j = j - 1
                        while temp_j > 0 and label[temp_j] == 'ADV':
                            temp_j -= 1
                        if label[temp_j] != 'SBV':
                            sum_sub += 1
                if sum_sub != 0:
                    i_sub = 0
                    tar_sub = random.randint(1,sum_sub)
                    new_line = {}
                    labels = []
                    add_sub = ''
                    flag = 0
                    source = cls
                    target = cls + line + sep
                    labels.append('correct')
                    for j in range(len(cws)):
                        if pos[j] == 'v' and (label[j] == 'HED' or label[j] == 'COO'):
                            temp_j = j - 1
                            while temp_j > 0 and label[temp_j] == 'ADV':
                                temp_j -= 1
                            if label[temp_j] != 'SBV':
                                i_sub += 1
                                add_index = temp_j
                            if i_sub == tar_sub:
                                temp_index = head[j] - 1
                                while label[temp_index] == 'COO':
                                    temp_index = head[temp_index] - 1
                                for m in range(len(cws)):
                                    if label[m] == 'SBV' and head[m] == temp_index + 1:
                                        flag = 1
                                        add_sub = cws[m]
                    if flag == 1:
                        for j in range(len(cws)):
                            if j == add_index:
                                for word in cws[j]:
                                    source += word
                                    labels.append('correct')
                                for word in add_sub:
                                    source += word
                                    labels.append('redu_sub')
                            else:
                                for word in cws[j]:
                                    source += word
                                    labels.append('correct')
                        source += sep
                        labels.append('correct')
                        new_line['source'] = source
                        new_line['target'] = target
                        new_line['label'] = labels
                        new_lines.append(new_line)
                        n += 1
    with open("gen_9_label.json",'w',encoding='utf-8') as f:
        json.dump(new_lines,f,ensure_ascii=False,indent=1)

