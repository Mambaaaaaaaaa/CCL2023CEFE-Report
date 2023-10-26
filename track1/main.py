# from transformers import BertForTokenClassification, set_seed 
from transformers import ElectraForTokenClassification, set_seed 
from torch.utils.data import DataLoader, Dataset
import torch
# from transformers import BertTokenizer
from transformers import ElectraTokenizer
import json
from torch.optim import AdamW
import logging
from tqdm import tqdm
import numpy as np

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

train_path = './gen_data/train.json'
val_path = './gen_data/val.json'

test_path = './gen_data/val.json'

checkpoint_dir = './output/checkpoint.tar'
max_seq_length = 128
LR = 5e-5
batch_size = 32
num_epochs = 50
set_seed = 1
TEST = False

class JsonDataset(Dataset):
    def __init__(self, mode):
        if mode == 'train':
            self.json_data = json.load(open(train_path, 'r'))
        else:
            self.json_data = json.load(open(val_path, 'r'))
    
    def __getitem__(self, idx):
        inputs = self.json_data[idx]['source'].strip()
        targets = self.json_data[idx]['label']
        inputs = inputs[5:-5]
        if len(inputs) > max_seq_length - 2:
            inputs = inputs[:(max_seq_length-2)]
            targets = targets[:max_seq_length-1]
            targets = targets + ['correct']
        _inputs = []
        _inputs.append("[CLS]")
        _inputs.extend(list(inputs))
        _inputs.append("[SEP]")

        # print(targets)
        if len(_inputs) != len(targets):
            print(len(_inputs))
            print(len(targets))
            print(targets)
        assert len(_inputs) == len(targets)
        input_ids = tokenizer.convert_tokens_to_ids(_inputs)
        input_mask = [1] * len(input_ids)
        label_ids = [labels_to_ids[x] for x in targets]
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            label_ids.append(-100)
            input_mask.append(0)
        
        assert len(input_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(input_mask) == max_seq_length

        return {'source': torch.LongTensor(input_ids), 
                'attention_mask': torch.LongTensor(input_mask), 
                'label': torch.LongTensor(label_ids)}
    
    def __len__(self):
        return len(self.json_data)

id_list = []
sent_list = []
gold_C_list = []   
gold_F_list = []

class JsonDataset_test(Dataset):
    def __init__(self):
        self.json_data = json.load(open(test_path, 'r'))
    
    def __getitem__(self, idx):
        temp_id = self.json_data[idx]['sent_id']
        id_list.append(temp_id)

        inputs = self.json_data[idx]['sent'].strip()
        sent_list.append(inputs)

        gold_C_inputs = self.json_data[idx]['CourseGrainedErrorType']
        gold_C_list.append(gold_C_inputs)

        if len(inputs) > max_seq_length - 2:
            inputs = inputs[:(max_seq_length-2)]
        _inputs = []
        _inputs.append("[CLS]")
        _inputs.extend(list(inputs))
        _inputs.append("[SEP]")

        input_ids = tokenizer.convert_tokens_to_ids(_inputs)
        input_mask = [1] * len(input_ids)
        label_ids = [1] * max_seq_length
        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
        
        assert len(input_ids) == max_seq_length
        assert len(label_ids) == max_seq_length
        assert len(input_mask) == max_seq_length

        return {'source': torch.LongTensor(input_ids), 
                'attention_mask': torch.LongTensor(input_mask), 
                'label': torch.LongTensor(label_ids)}
    
    def __len__(self):
        return len(self.json_data)


test_score = []
def test1():
    
    predict_Ctype = {'字符级错误':0,'成分残缺型错误':0,'成分赘余型错误':0,'成分搭配不当型错误':0}
    # predict_Ftype = {'缺字漏字':0,'错别字错误':0,'缺少标点':0,'错用标点':0,'主语不明':0,'谓语残缺':0,'宾语残缺':0,'其他成分残缺':0,'主语多余':0,'虚词多余':0,'其他成分多余':0,'语序不当':0,'动宾搭配不当':0,'其他搭配不当':0,}
    gold_Ctype = {'字符级错误':0,'成分残缺型错误':0,'成分赘余型错误':0,'成分搭配不当型错误':0}
    # gold_Ftype = {'缺字漏字':0,'错别字错误':0,'缺少标点':0,'错用标点':0,'主语不明':0,'谓语残缺':0,'宾语残缺':0,'其他成分残缺':0,'主语多余':0,'虚词多余':0,'其他成分多余':0,'语序不当':0,'动宾搭配不当':0,'其他搭配不当':0,}
    correct_Ctype = {'字符级错误':0,'成分残缺型错误':0,'成分赘余型错误':0,'成分搭配不当型错误':0}
    # correct_Ftype = {'缺字漏字':0,'错别字错误':0,'缺少标点':0,'错用标点':0,'主语不明':0,'谓语残缺':0,'宾语残缺':0,'其他成分残缺':0,'主语多余':0,'虚词多余':0,'其他成分多余':0,'语序不当':0,'动宾搭配不当':0,'其他搭配不当':0,}
    
    correct_pre = 0
    correct_gold = 0
    correct_cor = 0

    p_Ctype = {'字符级错误':0.0000,'成分残缺型错误':0.0000,'成分赘余型错误':0.0000,'成分搭配不当型错误':0.0000}
    # p_Ftype = {'缺字漏字':0.0000,'错别字错误':0.0000,'缺少标点':0.0000,'错用标点':0.0000,'主语不明':0.0000,'谓语残缺':0.0000,'宾语残缺':0.0000,'其他成分残缺':0.0000,'主语多余':0.0000,'虚词多余':0.0000,'其他成分多余':0.0000,'语序不当':0.0000,'动宾搭配不当':0.0000,'其他搭配不当':0.0000,}
    r_Ctype = {'字符级错误':0.0000,'成分残缺型错误':0.0000,'成分赘余型错误':0.0000,'成分搭配不当型错误':0.0000}
    # r_Ftype = {'缺字漏字':0.0000,'错别字错误':0.0000,'缺少标点':0.0000,'错用标点':0.0000,'主语不明':0.0000,'谓语残缺':0.0000,'宾语残缺':0.0000,'其他成分残缺':0.0000,'主语多余':0.0000,'虚词多余':0.0000,'其他成分多余':0.0000,'语序不当':0.0000,'动宾搭配不当':0.0000,'其他搭配不当':0.0000,}

    with torch.no_grad():
        model.eval()
        logger.info(('train on epoch %d' % (epoch_size+1)))
        test_dataset = JsonDataset_test()
        test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)

        t = tqdm(test_dataloader, desc=f'epoch{epoch_size + 1}', ncols=80)
        
        temp_i = 0
        for step, batch in enumerate(t):
            batch = {k: v.long().to(device) for k, v in batch.items()}
            res = model(  input_ids=batch['source'],
                            attention_mask=batch['attention_mask'],
                            labels=batch['label'])
            # predict = torch.argmax(res.logits, dim=-1).tolist()
            predict = torch.argmax(res.logits, dim=-1)
            predict[batch['attention_mask']==0] = -100
            predict = predict.tolist()
            

            for pre_labels in predict:
                
                pre_label = [ids_to_labels[i] for i in pre_labels if i != -100]

                temp_pre = {}
                temp_pre['CourseGrainedErrorType'] = []

                temp_gold = {}
                temp_gold['CourseGrainedErrorType'] = gold_C_list[temp_i]


                for label in pre_label:
                    if label == 'redu':
                        if '成分赘余型错误' not in temp_pre['CourseGrainedErrorType']:
                            predict_Ctype['成分赘余型错误'] += 1
                            temp_pre['CourseGrainedErrorType'].append('成分赘余型错误')
                    elif label == 'coll':   
                        if '成分搭配不当型错误' not in temp_pre['CourseGrainedErrorType']:
                            predict_Ctype['成分搭配不当型错误'] += 1
                            temp_pre['CourseGrainedErrorType'].append('成分搭配不当型错误')
                    elif label == 'miss':
                        if '成分残缺型错误' not in temp_pre['CourseGrainedErrorType']:
                            predict_Ctype['成分残缺型错误'] += 1
                            temp_pre['CourseGrainedErrorType'].append('成分残缺型错误')
                        
                if pre_label == ['correct'] * len(pre_label):
                    correct_pre += 1
                    if temp_gold['CourseGrainedErrorType'] == []:
                        correct_cor += 1
                if temp_gold['CourseGrainedErrorType'] == []:
                    correct_gold += 1

                for label in temp_gold['CourseGrainedErrorType']:
                    gold_Ctype[label] += 1
                # for label in temp_gold['FineGrainedErrorType']:
                #     gold_Ftype[label] += 1

                for pre_C in temp_pre['CourseGrainedErrorType']:
                    if pre_C in temp_gold['CourseGrainedErrorType']:
                        correct_Ctype[pre_C] += 1
                # for pre_F in temp_pre['FineGrainedErrorType']:
                #     if pre_F in temp_gold['FineGrainedErrorType']:
                #         correct_Ftype[pre_F] += 1
                
                temp_i += 1

    for c_Ctype in correct_Ctype:
        if gold_Ctype[c_Ctype] != 0:
            r_Ctype[c_Ctype] = '%.4f'%(correct_Ctype[c_Ctype]/gold_Ctype[c_Ctype])
        if predict_Ctype[c_Ctype] != 0:
            p_Ctype[c_Ctype] = '%.4f'%(correct_Ctype[c_Ctype]/predict_Ctype[c_Ctype])
    
    # for c_Ftype in correct_Ftype:
    #     if gold_Ftype[c_Ftype] != 0:
    #         r_Ftype[c_Ftype] = '%.4f'%(correct_Ftype[c_Ftype]/gold_Ftype[c_Ftype])
    #     if predict_Ftype[c_Ftype] != 0:
    #         p_Ftype[c_Ftype] = '%.4f'%(correct_Ftype[c_Ftype]/predict_Ftype[c_Ftype])

    print('predict')
    sum_c_p = 0
    sum_f_p = 0
    for key, value in predict_Ctype.items():
        sum_c_p += int(value)
        print(key, value)
    sum_c_p += correct_pre
    print('sum_predict_C','   ',sum_c_p)
    # for key, value in predict_Ftype.items():
    #     sum_f_p += int(value)
    #     print(key, value)
    # sum_f_p += correct_pre
    # print('sum_predict_F','   ',sum_f_p)

    print('-----------------------------------------------')
    print('gold')
    sum_c_g = 0
    sum_f_g = 0
    for key, value in gold_Ctype.items():
        sum_c_g += int(value)
        print(key, value)
    sum_c_g += correct_gold
    print('sum_gold_C','   ',sum_c_g)
    # for key, value in gold_Ftype.items():
    #     sum_f_g += int(value)
    #     print(key, value)
    # sum_f_g += correct_gold
    # print('sum_gold_F','   ',sum_f_g)

    print('-----------------------------------------------')
    print('correct')
    sum_c = 0
    sum_f = 0
    for key, value in correct_Ctype.items():
        sum_c += int(value)
        print(key, value)
    sum_c += correct_cor
    print('sum_correct_C','   ',sum_c)
    # for key, value in correct_Ftype.items():
    #     sum_f += int(value)
    #     print(key, value)
    # sum_f += correct_cor
    # print('sum_correct_F','   ',sum_f)

    print('-----------------------------------------------')
    print('precision     recall      F1')
    print('CourseGrainedErrorType:')

    C_f = []
    for key in p_Ctype.keys():
        p = float(p_Ctype[key])
        r = float(r_Ctype[key])
        if p + r == 0:
            f = 0
        else:
            f = 2 * p * r /(p + r)
        C_f.append(f)
        print(key,'    ',p,'   ',r,'    ',f)
    correct_p = correct_cor / correct_pre if correct_pre != 0 else 0
    correct_r = correct_cor / correct_gold if correct_gold != 0 else 0
    correct_f = 2 * correct_p * correct_r / (correct_p + correct_r) if correct_p + correct_r != 0 else 0
    C_f.append(correct_f)
    print('正确','    ',correct_p,'   ',correct_r,'    ',correct_f)

    # print('FineGrainedErrorType:')
    # F_f = []
    # for key in p_Ftype.keys():
    #     p = float(p_Ftype[key])
    #     r = float(r_Ftype[key])
    #     if p + r == 0:
    #         f = 0
    #     else:
    #         f = 2 * p * r /(p + r)
    #     F_f.append(f)
    #     print(key,'    ',p,'   ',r,'    ',f)
    # F_f.append(correct_f)
    # print('正确','    ',correct_p,'   ',correct_r,'    ',correct_f)

    print('-----------------------------------------------')

    
    C_micro_p = sum_c / sum_c_p if sum_c_p != 0 else 0
    C_micro_r = sum_c / sum_c_g if sum_c_g != 0 else 0
    # F_micro_p = sum_f / sum_f_p if sum_f_p != 0 else 0
    # F_micro_r = sum_f / sum_f_g if sum_f_g != 0 else 0

    C_micro_f = 2 * C_micro_p * C_micro_r / (C_micro_p + C_micro_r) if C_micro_p + C_micro_r != 0 else 0
    # F_micro_f = 2 * F_micro_p * F_micro_r / (F_micro_p + F_micro_r) if F_micro_p + F_micro_r != 0 else 0

    score = C_micro_f

    C_macro_f = sum(C_f) / 5
    # F_macro_f = sum(F_f) / 15

    print('CourseGrainedErrorType_micro:','    ','p:',C_micro_p,'    ','r:',C_micro_r,'    ','f:',C_micro_f)
    # print('FineGrainedErrorType_micro:','    ','p:',F_micro_p,'    ','r:',F_micro_r,'    ','f:',F_micro_f)
    print('CourseGrainedErrorType_macro_f:',C_macro_f)
    # print('FineGrainedErrorTypemacro_f:',F_macro_f)
    print('final_score:',score)

    test_score.append(score)
    return score
        
train_score = []
def test2():
    
    predict_Ctype = {'字符级错误':0,'成分残缺型错误':0,'成分赘余型错误':0,'成分搭配不当型错误':0}
    # predict_Ftype = {'缺字漏字':0,'错别字错误':0,'缺少标点':0,'错用标点':0,'主语不明':0,'谓语残缺':0,'宾语残缺':0,'其他成分残缺':0,'主语多余':0,'虚词多余':0,'其他成分多余':0,'语序不当':0,'动宾搭配不当':0,'其他搭配不当':0,}
    gold_Ctype = {'字符级错误':0,'成分残缺型错误':0,'成分赘余型错误':0,'成分搭配不当型错误':0}
    # gold_Ftype = {'缺字漏字':0,'错别字错误':0,'缺少标点':0,'错用标点':0,'主语不明':0,'谓语残缺':0,'宾语残缺':0,'其他成分残缺':0,'主语多余':0,'虚词多余':0,'其他成分多余':0,'语序不当':0,'动宾搭配不当':0,'其他搭配不当':0,}
    correct_Ctype = {'字符级错误':0,'成分残缺型错误':0,'成分赘余型错误':0,'成分搭配不当型错误':0}
    # correct_Ftype = {'缺字漏字':0,'错别字错误':0,'缺少标点':0,'错用标点':0,'主语不明':0,'谓语残缺':0,'宾语残缺':0,'其他成分残缺':0,'主语多余':0,'虚词多余':0,'其他成分多余':0,'语序不当':0,'动宾搭配不当':0,'其他搭配不当':0,}
    
    correct_pre = 0
    correct_gold = 0
    correct_cor = 0

    p_Ctype = {'字符级错误':0.0000,'成分残缺型错误':0.0000,'成分赘余型错误':0.0000,'成分搭配不当型错误':0.0000}
    # p_Ftype = {'缺字漏字':0.0000,'错别字错误':0.0000,'缺少标点':0.0000,'错用标点':0.0000,'主语不明':0.0000,'谓语残缺':0.0000,'宾语残缺':0.0000,'其他成分残缺':0.0000,'主语多余':0.0000,'虚词多余':0.0000,'其他成分多余':0.0000,'语序不当':0.0000,'动宾搭配不当':0.0000,'其他搭配不当':0.0000,}
    r_Ctype = {'字符级错误':0.0000,'成分残缺型错误':0.0000,'成分赘余型错误':0.0000,'成分搭配不当型错误':0.0000}
    
    with torch.no_grad():
        model.eval()
        logger.info(('train on epoch %d' % (epoch_size+1)))

        t = tqdm(train_dataloader, desc=f'epoch{epoch_size + 1}', ncols=80)
        
        temp_i = 0
        for step, batch in enumerate(t):
            batch = {k: v.long().to(device) for k, v in batch.items()}
            res = model(  input_ids=batch['source'],
                            attention_mask=batch['attention_mask'],
                            labels=batch['label'])
            # predict = torch.argmax(res.logits, dim=-1).tolist()
            predict = torch.argmax(res.logits, dim=-1)
            predict[batch['attention_mask']==0] = -100
            predict = predict.tolist()
            gold = batch['label'].tolist()

            for pre_labels,gold_labels in zip(predict,gold):
                
                pre_label = [ids_to_labels[i] for i in pre_labels if i != -100]
                gold_label = [ids_to_labels[i] for i in gold_labels if i != -100]

                temp_pre = {}
                temp_pre['CourseGrainedErrorType'] = []
                # temp_pre['FineGrainedErrorType'] = []
                temp_gold = {}
                temp_gold['CourseGrainedErrorType'] = []
                # temp_gold['FineGrainedErrorType'] = []

                for label in pre_label:
                    if label == 'redu':
                        if '成分赘余型错误' not in temp_pre['CourseGrainedErrorType']:
                            predict_Ctype['成分赘余型错误'] += 1
                            temp_pre['CourseGrainedErrorType'].append('成分赘余型错误')
                    elif label == 'coll':   
                        if '成分搭配不当型错误' not in temp_pre['CourseGrainedErrorType']:
                            predict_Ctype['成分搭配不当型错误'] += 1
                            temp_pre['CourseGrainedErrorType'].append('成分搭配不当型错误')
                    elif label == 'miss':
                        if '成分残缺型错误' not in temp_pre['CourseGrainedErrorType']:
                            predict_Ctype['成分残缺型错误'] += 1
                            temp_pre['CourseGrainedErrorType'].append('成分残缺型错误')
                
                for label in gold_label:
                    if label == 'redu':
                        if '成分赘余型错误' not in temp_gold['CourseGrainedErrorType']:
                            gold_Ctype['成分赘余型错误'] += 1
                            temp_gold['CourseGrainedErrorType'].append('成分赘余型错误')
                    elif label == 'coll':   
                        if '成分搭配不当型错误' not in temp_gold['CourseGrainedErrorType']:
                            gold_Ctype['成分搭配不当型错误'] += 1
                            temp_gold['CourseGrainedErrorType'].append('成分搭配不当型错误')
                    elif label == 'miss':
                        if '成分残缺型错误' not in temp_gold['CourseGrainedErrorType']:
                            gold_Ctype['成分残缺型错误'] += 1
                            temp_gold['CourseGrainedErrorType'].append('成分残缺型错误')
                

                if pre_label == ['correct'] * len(pre_label):
                    correct_pre += 1
                    if temp_gold['CourseGrainedErrorType'] == []:
                        correct_cor += 1
                if temp_gold['CourseGrainedErrorType'] == []:
                    correct_gold += 1

                for label in temp_gold['CourseGrainedErrorType']:
                    gold_Ctype[label] += 1
                # for label in temp_gold['FineGrainedErrorType']:
                #     gold_Ftype[label] += 1

                for pre_C in temp_pre['CourseGrainedErrorType']:
                    if pre_C in temp_gold['CourseGrainedErrorType']:
                        correct_Ctype[pre_C] += 1
                # for pre_F in temp_pre['FineGrainedErrorType']:
                #     if pre_F in temp_gold['FineGrainedErrorType']:
                #         correct_Ftype[pre_F] += 1
                
                temp_i += 1

    for c_Ctype in correct_Ctype:
        if gold_Ctype[c_Ctype] != 0:
            r_Ctype[c_Ctype] = '%.4f'%(correct_Ctype[c_Ctype]/gold_Ctype[c_Ctype])
        if predict_Ctype[c_Ctype] != 0:
            p_Ctype[c_Ctype] = '%.4f'%(correct_Ctype[c_Ctype]/predict_Ctype[c_Ctype])
    
    # for c_Ftype in correct_Ftype:
    #     if gold_Ftype[c_Ftype] != 0:
    #         r_Ftype[c_Ftype] = '%.4f'%(correct_Ftype[c_Ftype]/gold_Ftype[c_Ftype])
    #     if predict_Ftype[c_Ftype] != 0:
    #         p_Ftype[c_Ftype] = '%.4f'%(correct_Ftype[c_Ftype]/predict_Ftype[c_Ftype])

    print('predict')
    sum_c_p = 0
    sum_f_p = 0
    for key, value in predict_Ctype.items():
        sum_c_p += int(value)
        print(key, value)
    sum_c_p += correct_pre
    print('sum_predict_C','   ',sum_c_p)
    # for key, value in predict_Ftype.items():
    #     sum_f_p += int(value)
    #     print(key, value)
    # sum_f_p += correct_pre
    # print('sum_predict_F','   ',sum_f_p)

    print('-----------------------------------------------')
    print('gold')
    sum_c_g = 0
    sum_f_g = 0
    for key, value in gold_Ctype.items():
        sum_c_g += int(value)
        print(key, value)
    sum_c_g += correct_gold
    print('sum_gold_C','   ',sum_c_g)
    # for key, value in gold_Ftype.items():
    #     sum_f_g += int(value)
    #     print(key, value)
    # sum_f_g += correct_gold
    # print('sum_gold_F','   ',sum_f_g)

    print('-----------------------------------------------')
    print('correct')
    sum_c = 0
    sum_f = 0
    for key, value in correct_Ctype.items():
        sum_c += int(value)
        print(key, value)
    sum_c += correct_cor
    print('sum_correct_C','   ',sum_c)
    # for key, value in correct_Ftype.items():
    #     sum_f += int(value)
    #     print(key, value)
    # sum_f += correct_cor
    # print('sum_correct_F','   ',sum_f)

    print('-----------------------------------------------')
    print('precision     recall      F1')
    print('CourseGrainedErrorType:')

    C_f = []
    for key in p_Ctype.keys():
        p = float(p_Ctype[key])
        r = float(r_Ctype[key])
        if p + r == 0:
            f = 0
        else:
            f = 2 * p * r /(p + r)
        C_f.append(f)
        print(key,'    ',p,'   ',r,'    ',f)
    correct_p = correct_cor / correct_pre if correct_pre != 0 else 0
    correct_r = correct_cor / correct_gold if correct_gold != 0 else 0
    correct_f = 2 * correct_p * correct_r / (correct_p + correct_r) if correct_p + correct_r != 0 else 0
    C_f.append(correct_f)
    print('正确','    ',correct_p,'   ',correct_r,'    ',correct_f)

    # print('FineGrainedErrorType:')
    # F_f = []
    # for key in p_Ftype.keys():
    #     p = float(p_Ftype[key])
    #     r = float(r_Ftype[key])
    #     if p + r == 0:
    #         f = 0
    #     else:
    #         f = 2 * p * r /(p + r)
    #     F_f.append(f)
    #     print(key,'    ',p,'   ',r,'    ',f)
    # F_f.append(correct_f)
    # print('正确','    ',correct_p,'   ',correct_r,'    ',correct_f)

    print('-----------------------------------------------')

    
    C_micro_p = sum_c / sum_c_p if sum_c_p != 0 else 0
    C_micro_r = sum_c / sum_c_g if sum_c_g != 0 else 0
    # F_micro_p = sum_f / sum_f_p if sum_f_p != 0 else 0
    # F_micro_r = sum_f / sum_f_g if sum_f_g != 0 else 0

    C_micro_f = 2 * C_micro_p * C_micro_r / (C_micro_p + C_micro_r) if C_micro_p + C_micro_r != 0 else 0
    # F_micro_f = 2 * F_micro_p * F_micro_r / (F_micro_p + F_micro_r) if F_micro_p + F_micro_r != 0 else 0

    score = C_micro_f

    C_macro_f = sum(C_f) / 5
    # F_macro_f = sum(F_f) / 15

    print('CourseGrainedErrorType_micro:','    ','p:',C_micro_p,'    ','r:',C_micro_r,'    ','f:',C_micro_f)
    # print('FineGrainedErrorType_micro:','    ','p:',F_micro_p,'    ','r:',F_micro_r,'    ','f:',F_micro_f)
    print('CourseGrainedErrorType_macro_f:',C_macro_f)
    # print('FineGrainedErrorTypemacro_f:',F_macro_f)
    print('final_score:',score)
    
    train_score.append(score)
    return score


if __name__ == '__main__':
    # data
    unique_labels = {'correct','redu','coll','miss',}
    labels_to_ids = {k: v for v, k in enumerate(sorted(unique_labels))}
    ids_to_labels = {v: k for v, k in enumerate(sorted(unique_labels))}
    

    tokenizer = ElectraTokenizer.from_pretrained('hfl/chinese-electra-180g-base-discriminator', cache_dir='./pt_model_elec')

    train_dataset = JsonDataset('train')
    # test_dataset = JsonDataset('val')
    
    # model

    model = ElectraForTokenClassification.from_pretrained('hfl/chinese-electra-180g-base-discriminator', cache_dir='./pt_model_elec', num_labels=len(unique_labels))
    
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=LR, eps=1e-8)

    epoch_size = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model.to(device)
    
    train_dataloader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
    # test_dataloader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)
    
    best_score = -1
    best_epoch = -1
    

    if TEST:
        model.load_state_dict(torch.load(checkpoint_dir, map_location=torch.device('cpu')), strict=False)
        test1()
        exit()

    for epoch_size in range(num_epochs):
        model.train()
        logger.info(('train on epoch %d' % (epoch_size+1)))
        t = tqdm(train_dataloader, desc=f'epoch{epoch_size + 1}', ncols=80)
        for step, batch in enumerate(t):
            batch = {k: v.long().to(device) for k, v in batch.items()}
            res = model(  input_ids=batch['source'],
                            attention_mask=batch['attention_mask'],
                            labels=batch['label'])
            predict = torch.argmax(res.logits, dim=-1)
            predict[batch['label']==1] = -1
            predict = (predict == batch['label']).long()
            acc = torch.sum(torch.masked_select(predict, batch['attention_mask']==1))/torch.sum(batch['label'] != 1)
            
            res.loss.backward()
            t.set_postfix_str('loss ={:^7.6f}&{:.3f}'.format(res.loss.item(), acc.item()))
            t.update()
            optimizer.step() 
            optimizer.zero_grad()
            
        print('Train Set:')
        test2()
        print('Test Set:')
        cur_score = test1()
        if cur_score > best_score:
            best_score = cur_score
            best_epoch = epoch_size + 1
            torch.save(model.state_dict(), checkpoint_dir)

    print('best_score:',best_score)
    print('best_epoch:',best_epoch)
        
    print('Train Set:')
    for i,j in enumerate(train_score):
        print(i + 1,'   ',j)

    print('Test Set:')
    for i,j in enumerate(test_score):
        print(i + 1,'   ',j)