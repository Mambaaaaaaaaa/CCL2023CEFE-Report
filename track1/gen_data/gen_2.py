
import random
from opencc import OpenCC
import json


cc = OpenCC('t2s')



if __name__ == '__main__':
    n = 0
    i = 0
    cls = '[CLS]'
    sep = '[SEP]'
    new_lines = []
    confuse_set = {}
    
    with open('../source_data/token_set.txt', 'r',encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            temp_line =  line.strip('\n').split('\t')
            ch = temp_line[0]
            if cc.convert(ch) == ch:
                confuse_set[ch] = []
                for word in temp_line[1]:
                    new_word = cc.convert(word)
                    if word == new_word:
                        confuse_set[temp_line[0]].append(word)
    with open('../source_data/new_cofuse_set.txt', 'w',encoding='utf-8') as f:
        for word in confuse_set:
            f.write(word)
            f.write('\t')
            for confuse_word in confuse_set[word]:
                f.write(confuse_word)
            f.write('\n')
    with open('../source_data/new_train.txt', 'r',encoding='utf-8') as f:
        lines = f.readlines()
        while n != 1000:
            i += 1
            if i % 10 == 5:
                line = lines[i].strip('\n')
                line_len = len(line)
                confuse_index = random.randint(0,line_len-1)
                while confuse_set.get(line[confuse_index]) == None:
                    confuse_index = random.randint(0,line_len-1)
                new_line = {}
                label = []
                source = cls
                target = cls + line + sep
                label.append('correct')
                for j in range(line_len):
                    if j == confuse_index:
                        confuse_word_set = confuse_set[line[j]]
                        new_index = random.randint(0,len(confuse_word_set)-1)
                        source += confuse_word_set[new_index]
                        label.append('char_error')
                    else:
                        source += line[j]
                        label.append('correct')
                source += sep
                label.append('correct')
                new_line['source'] = source
                new_line['target'] = target
                new_line['label'] = label
                new_lines.append(new_line)
                n += 1
    
    with open("gen_2_label.json",'w',encoding='utf-8') as f:
        json.dump(new_lines,f,ensure_ascii=False,indent=1)


