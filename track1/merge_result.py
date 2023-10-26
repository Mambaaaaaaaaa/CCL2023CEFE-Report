
import json
from tqdm import tqdm

# source_path = './part_testA_gold.json'
# predict_all_path = 'testA_0609.json'
# predict_char_path = 'testA_0608_char_1.json'
# predict_coll_path = 'testA_0609_coll.json'
# predict_miss_path = 'testA_0609_miss.json'
# predict_redu_path = 'testA_0609_redu.json'
# result_path = 'A_0609_3.json'

source_path = './testB.json'
predict_all_path = 'testB_0609.json'
predict_char_path = 'testB_0608_char.json'
predict_coll_path = 'testB_0609_coll.json'
predict_miss_path = 'testB_0609_miss.json'
predict_redu_path = 'testB_0609_redu.json'
result_path = 'track1_0609.json'


source = []


with open(source_path,'r',encoding='utf-8') as f:
    string = f.read()
    raw_data = json.loads(string)
    for line in raw_data:
        id = line['sent_id']
        sent = line['sent']

        temp_line = {}
        temp_line['sent_id'] = id
        temp_line['sent'] = sent
        temp_line['CourseGrainedErrorType'] = []
        temp_line['FineGrainedErrorType'] = []

        source.append(temp_line)

temp_all = []

with open(predict_all_path,'r',encoding='utf-8') as f:
    string = f.read()
    raw_data = json.loads(string)
    for line in raw_data:
        id = line['sent_id']
        sent = line['sent']
        labels = line['label']


        temp_line = {}
        temp_line['sent_id'] = id
        temp_line['sent'] = sent
        temp_line['CourseGrainedErrorType'] = []

        for label in labels:
            if label == 'redu':
                if '成分赘余型错误' not in temp_line['CourseGrainedErrorType']:
                    temp_line['CourseGrainedErrorType'].append('成分赘余型错误')
            elif label == 'coll':   
                if '成分搭配不当型错误' not in temp_line['CourseGrainedErrorType']:
                    temp_line['CourseGrainedErrorType'].append('成分搭配不当型错误')
            elif label == 'miss':
                if '成分残缺型错误' not in temp_line['CourseGrainedErrorType']:
                    temp_line['CourseGrainedErrorType'].append('成分残缺型错误')

        temp_all.append(temp_line)

temp_char = []

with open(predict_char_path,'r',encoding='utf-8') as f:
    string = f.read()
    raw_data = json.loads(string)
    for line in raw_data:
        id = line['sent_id']
        sent = line['sent']
        labels = line['label']

        temp_line = {}
        temp_line['sent_id'] = id
        temp_line['sent'] = sent
        temp_line['CourseGrainedErrorType'] = []
        temp_line['FineGrainedErrorType'] = []


        for label in labels:
            if label == 'char_append':
                if '字符级错误' not in temp_line['CourseGrainedErrorType']:
                    temp_line['CourseGrainedErrorType'].append('字符级错误')
                if '缺字漏字' not in temp_line['FineGrainedErrorType']:
                    temp_line['FineGrainedErrorType'].append('缺字漏字')
            elif label == 'char_error' or label == 'char_delete':
                if '字符级错误' not in temp_line['CourseGrainedErrorType']:
                    temp_line['CourseGrainedErrorType'].append('字符级错误')
                if '错别字错误' not in temp_line['FineGrainedErrorType']:
                    temp_line['FineGrainedErrorType'].append('错别字错误')
            elif label == 'char_punc_append':
                if '字符级错误' not in temp_line['CourseGrainedErrorType']:
                    temp_line['CourseGrainedErrorType'].append('字符级错误')
                if '缺少标点' not in temp_line['FineGrainedErrorType']:
                    temp_line['FineGrainedErrorType'].append('缺少标点')
            elif label == 'char_punc_error'or label == 'char_punc_delete':
                if '字符级错误' not in temp_line['CourseGrainedErrorType']:
                    temp_line['CourseGrainedErrorType'].append('字符级错误')
                if '错用标点' not in temp_line['FineGrainedErrorType']:
                    temp_line['FineGrainedErrorType'].append('错用标点')


        temp_char.append(temp_line)




temp_coll = []

with open(predict_coll_path,'r',encoding='utf-8') as f:
    string = f.read()
    raw_data = json.loads(string)
    temp_index = 0
    for line in raw_data:
        id = line['sent_id']
        sent = line['sent']
        labels = line['label']

        temp_line = {}
        temp_line['sent_id'] = id
        temp_line['sent'] = sent
        temp_line['CourseGrainedErrorType'] = []
        temp_line['FineGrainedErrorType'] = []

        temp_label = temp_all[temp_index]['CourseGrainedErrorType']

        if '成分搭配不当型错误' in temp_label:
            for label in labels:
                if label == 'coll_order':
                    if '成分搭配不当型错误' not in temp_line['CourseGrainedErrorType']:
                        temp_line['CourseGrainedErrorType'].append('成分搭配不当型错误')
                    if '语序不当' not in temp_line['FineGrainedErrorType']:
                        temp_line['FineGrainedErrorType'].append('语序不当')
                elif label == 'coll_vobj':
                    if '成分搭配不当型错误' not in temp_line['CourseGrainedErrorType']:
                        temp_line['CourseGrainedErrorType'].append('成分搭配不当型错误')
                    if '动宾搭配不当' not in temp_line['FineGrainedErrorType']:
                        temp_line['FineGrainedErrorType'].append('动宾搭配不当')
               
        temp_index += 1
        temp_coll.append(temp_line)

temp_miss = []

with open(predict_miss_path,'r',encoding='utf-8') as f:
    string = f.read()
    raw_data = json.loads(string)
    temp_index = 0
    for line in raw_data:
        id = line['sent_id']
        sent = line['sent']
        labels = line['label']

        temp_line = {}
        temp_line['sent_id'] = id
        temp_line['sent'] = sent
        temp_line['CourseGrainedErrorType'] = []
        temp_line['FineGrainedErrorType'] = []

        temp_label = temp_all[temp_index]['CourseGrainedErrorType']


        if '成分残缺型错误' in temp_label:
            for label in labels:
                if label == 'miss_sub':
                    if '成分残缺型错误' not in temp_line['CourseGrainedErrorType']:
                        temp_line['CourseGrainedErrorType'].append('成分残缺型错误')
                    if '主语不明' not in temp_line['FineGrainedErrorType']:
                        temp_line['FineGrainedErrorType'].append('主语不明')
                

        temp_index += 1
        temp_miss.append(temp_line)


temp_redu = []

with open(predict_redu_path,'r',encoding='utf-8') as f:
    string = f.read()
    raw_data = json.loads(string)
    temp_index = 0
    for line in raw_data:
        id = line['sent_id']
        sent = line['sent']
        labels = line['label']

        temp_line = {}
        temp_line['sent_id'] = id
        temp_line['sent'] = sent
        temp_line['CourseGrainedErrorType'] = []
        temp_line['FineGrainedErrorType'] = []

        temp_label = temp_all[temp_index]['CourseGrainedErrorType']



        if 'redu_other' in labels:
            redu_other = []
            start_index = []
            end_index = []
            new_labels = labels[1:-1]
            sen = sent
            temp_j = 0
            while temp_j < len(sen):
                if new_labels[temp_j] != 'redu_other':
                    while temp_j < len(sen) and new_labels[temp_j] != 'redu_other':
                        temp_j += 1
                else:
                    start = temp_j
                    while new_labels[temp_j] == 'redu_other':
                        temp_j += 1
                    end = temp_j - 1
                    word = sen[start:end+1]
                    redu_other.append(word)
                    start_index.append(start)
                    end_index.append(end)

            # print(sen)
            # print(redu_other)
            # print(start_index)
            # print(end_index)
        
            
            for i in range(len(redu_other)):
                if sen.count(redu_other[i]) == 1:
                    # print(redu_other[i])
                    for j in range(start_index[i]+1,end_index[i]+2):
                        # print(sen[j-1])
                        labels[j] = 'correct'


        if 'redu_other' in labels:
            for i in range(1,len(labels)-1):
                if labels[i] == 'redu_other' and labels[i-1] != 'redu_other' and labels[i+1] != 'redu_other':
                    labels[i] = 'correct'

        if '成分赘余型错误' in temp_label:
            for label in labels:

                if label == 'redu_emp':
                    if '成分赘余型错误' not in temp_line['CourseGrainedErrorType']:
                        temp_line['CourseGrainedErrorType'].append('成分赘余型错误')
                    if '虚词多余' not in temp_line['FineGrainedErrorType']:
                        temp_line['FineGrainedErrorType'].append('虚词多余')
                elif label == 'redu_other':
                    if '成分赘余型错误' not in temp_line['CourseGrainedErrorType']:
                        temp_line['CourseGrainedErrorType'].append('成分赘余型错误')
                    if '其他成分多余' not in temp_line['FineGrainedErrorType']:
                        temp_line['FineGrainedErrorType'].append('其他成分多余')
            
        
        temp_index += 1
        temp_redu.append(temp_line)


for i in range(len(source)):
    Ctype = temp_coll[i]['CourseGrainedErrorType'] + temp_miss[i]['CourseGrainedErrorType'] + temp_redu[i]['CourseGrainedErrorType'] + temp_char[i]['CourseGrainedErrorType']
    Ctype = list(set(Ctype))
    Ftype = temp_coll[i]['FineGrainedErrorType'] + temp_miss[i]['FineGrainedErrorType'] + temp_redu[i]['FineGrainedErrorType'] + temp_char[i]['FineGrainedErrorType']
    Ftype = list(set(Ftype))
    source[i]['CourseGrainedErrorType'] = Ctype
    source[i]['FineGrainedErrorType'] = Ftype


with open(result_path,'w',encoding='utf-8') as f:
    json.dump(source,f,ensure_ascii=False,indent=1)



