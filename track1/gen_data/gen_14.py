import json 
import random
from ltp import LTP

ltp = LTP()

if __name__ == '__main__':
    con_sub_v = {}
    with open('../source_data/confuse_sub_v.json', 'r',encoding='utf-8') as f:
        str = f.read()
        raw_data = json.loads(str)
        for sub in raw_data:
            con_sub_v[sub] = list(set(raw_data[sub]))
    con_v_sub = {}
    with open('../source_data/confuse_v_sub.json', 'r',encoding='utf-8') as f:
        str = f.read()
        raw_data = json.loads(str)
        for v in raw_data:
            con_v_sub[v] = list(set(raw_data[v]))
    n = 0
    i = 8000
    cls = '[CLS]'
    sep = '[SEP]'
    new_lines = []
    with open('../source_data/new_train.txt', 'r',encoding='utf-8') as f:
        lines = f.readlines()
        while n != 1000:
            i += 1
            if i % 10 == 1:
                line = lines[i].strip('\n')
                result =ltp.pipeline([line], tasks = ["cws","dep"])
                cws = result.cws[0]
                head = result.dep[0]['head']
                label = result.dep[0]['label']
                res = list(zip(cws,head,label))
                sum_sub_v = 0
                for (c,h,l) in res:
                    if l == 'SBV':
                        sum_sub_v += 1
                if sum_sub_v != 0:
                    i_sub_v = 0
                    tar_sub_v = random.randint(1,sum_sub_v)
                    new_line = {}
                    label = []
                    source = cls
                    target = cls + line + sep
                    label.append('correct')
                    flag = 0
                    temp_label = []
                    temp_source = []
                    for p in range(len(cws)):
                        temp_label.append([])
                        temp_source.append([])
                    temp_index = 0
                    for (c,h,l) in res:
                        if l == 'SBV':
                            i_sub_v += 1
                            if i_sub_v == tar_sub_v:
                                temp_i = random.randint(0,1)
                                v = cws[h-1]
                                sub = c
                                if temp_i == 1:
                                    if con_v_sub.get(v) != None:
                                        sub_set = con_v_sub[v]
                                        if sub in sub_set:
                                            sub_set.remove(sub)
                                        if len(sub_set) > 0:
                                            flag = 1
                                            index = random.randint(0,len(sub_set)-1)
                                            new_sub = sub_set[index]
                                            temp_source[temp_index] = []
                                            temp_label[temp_index] = []
                                            temp_label[h-1] = []
                                            temp_source[h-1] = []
                                            for word in new_sub:
                                                temp_source[temp_index].append(word)
                                                temp_label[temp_index].append('coll_other')
                                            for word in v:
                                                temp_source[h-1].append(word)
                                                temp_label[h-1].append('coll_other')
                                else:
                                    if con_sub_v.get(sub) != None:
                                        v_set = con_sub_v[sub]
                                        if v in v_set:
                                            v_set.remove(v)
                                        if len(v_set) > 0:
                                            flag = 1
                                            index = random.randint(0,len(v_set)-1)
                                            new_v = v_set[index]
                                            temp_source[temp_index] = []
                                            temp_label[temp_index] = []
                                            temp_label[h-1] = []
                                            temp_source[h-1] = []
                                            for word in new_v:
                                                temp_source[h-1].append(word)
                                                temp_label[h-1].append('coll_other')
                                            for word in sub:
                                                temp_source[temp_index].append(word)
                                                temp_label[temp_index].append('coll_other')
                            else:
                                if temp_source[temp_index] == []:
                                    for word in c :
                                        temp_source[temp_index].append(word)
                                        temp_label[temp_index].append('correct')
                        else:
                            if temp_source[temp_index] == []:
                                    for word in c :
                                        temp_source[temp_index].append(word)
                                        temp_label[temp_index].append('correct')
                        temp_index += 1


                    if flag == 1:
                        for j in range(len(cws)):
                            for word in temp_source[j]:
                                source += word
                            for temp_l in temp_label[j]:
                                label.append(temp_l)
                        source += sep
                        label.append('correct')
                        new_line['source'] = source
                        new_line['target'] = target
                        new_line['label'] = label
                        new_lines.append(new_line)
                        print(len(source)-len(label))
                        n += 1
    with open("gen_14_label.json",'w',encoding='utf-8') as f:
        json.dump(new_lines,f,ensure_ascii=False,indent=1)

                    