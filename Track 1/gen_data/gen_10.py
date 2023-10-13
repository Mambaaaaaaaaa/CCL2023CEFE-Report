import random
import json
from ltp import LTP

ltp = LTP()



if __name__ == '__main__':
    n = 0
    i = 4000
    cls = '[CLS]'
    sep = '[SEP]'
    new_lines = []
    with open('../source_data/new_train.txt', 'r',encoding='utf-8') as f:
        lines = f.readlines()
        while n != 1000:
            i += 1
            if i % 10 == 3:
                line = lines[i].strip('\n')
                result =ltp.pipeline([line], tasks = ["cws","dep","pos"])
                cws = result.cws[0]
                pos = result.pos[0]
                head = result.dep[0]['head']
                label = result.dep[0]['label']
                sum_emp = 0
                for j in range(len(cws)):
                    if (pos[j] == 'a' or pos[j] == 'd' or pos[j] == 'q')and j < len(cws) - 1:
                            if cws[j + 1] != '的' and cws[j + 1] != '地':
                                sum_emp += 1
                    elif pos[j] == 'v' and j > 0:
                            if cws[j -1] != '所':
                                sum_emp += 1
                if sum_emp != 0:
                    i_emp = 0
                    tar_emp = random.randint(1,sum_emp)
                    new_line = {}
                    label = []
                    source = cls
                    target = cls + line + sep
                    label.append('correct')
                    for j in range(len(cws)):
                        flag = 0
                        if (pos[j] == 'a' or pos[j] == 'd' or pos[j] == 'q' )and j < len(cws) - 1:
                                if cws[j + 1] != '的' and cws[j + 1] != '地':
                                    i_emp += 1
                                    flag = 1
                                else:
                                    for word in cws[j]:
                                        source += word
                                        label.append('correct')
                        elif pos[j] == 'v' and j > 0:
                                if cws[j -1] != '所':
                                    i_emp += 1
                                    flag = 1
                                else:
                                    for word in cws[j]:
                                        source += word
                                        label.append('correct')
                        else:
                            for word in cws[j]:
                                source += word
                                label.append('correct')
                        if flag == 1 and i_emp == tar_emp:
                            if pos[j] == 'a' or pos[j] == 'q':
                                for word in cws[j]:
                                    source += word
                                    label.append('correct')
                                source += '的'
                                label.append('redu_emp')
                            elif pos[j] == 'd' :
                                for word in cws[j]:
                                    source += word
                                    label.append('correct')
                                source += '地'
                                label.append('redu_emp')
                            elif pos[j] == 'v' :
                                source += '所'
                                label.append('redu_emp')
                                for word in cws[j]:
                                    source += word
                                    label.append('correct')
                                source += '的'
                                label.append('redu_emp')
                        elif flag == 1:
                            for word in cws[j]:
                                source += word
                                label.append('correct')
                    source += sep
                    label.append('correct')
                    new_line['source'] = source
                    new_line['target'] = target
                    new_line['label'] = label
                    new_lines.append(new_line)
                    n += 1
    with open("gen_10_label.json",'w',encoding='utf-8') as f:
        json.dump(new_lines,f,ensure_ascii=False,indent=1)

