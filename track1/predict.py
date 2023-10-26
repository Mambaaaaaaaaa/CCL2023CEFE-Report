from transformers import BertTokenizer
from transformers import ElectraTokenizer
import torch
from transformers import BertForTokenClassification
from transformers import ElectraForTokenClassification
import json
from torch.utils.data import Dataset,DataLoader
from tqdm import tqdm


# tokenizer = BertTokenizer.from_pretrained('bert-base-chinese', cache_dir='./example/pt_model')
tokenizer = ElectraTokenizer.from_pretrained('hfl/chinese-electra-180g-base-discriminator', cache_dir='./example/pt_model_elec')

unique_labels = {'correct','redu_sub','redu_emp','redu_other','coll_order','coll_vobj','coll_other','miss_sub','miss_pre','miss_obj','miss_other','char_append','char_error','char_punc_append','char_punc_error'}
labels_to_ids = {k: v for v, k in enumerate(sorted(unique_labels))}
ids_to_labels = {v: k for v, k in enumerate(sorted(unique_labels))}

max_seq_length = 128
batch_size = 32
id_list = []
sent_list = []


checkpoint_dir = './output/checkpoint.tar'
predict_path = './testB.json'
result_path = "testB_redu.json"


class JsonDataset(Dataset):
    def __init__(self):
        self.json_data = json.load(open(predict_path, 'r'))
    
    def __getitem__(self, idx):
        temp_id = self.json_data[idx]['sent_id']
        id_list.append(temp_id)
        inputs = self.json_data[idx]['sent'].strip()
        sent_list.append(inputs)
        if len(inputs) > max_seq_length - 2:
            inputs = inputs[:(max_seq_length-2)]
        _inputs = []
        _inputs.append("[CLS]")
        _inputs.extend(list(inputs))
        _inputs.append("[SEP]")

        input_ids = tokenizer.convert_tokens_to_ids(_inputs)
        input_mask = [1] * len(input_ids)
        label_ids = [7] * max_seq_length
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



# model = BertForTokenClassification.from_pretrained('bert-base-chinese', num_labels=len(unique_labels), cache_dir='./example/pt_model')
model = ElectraForTokenClassification.from_pretrained('hfl/chinese-electra-180g-base-discriminator', cache_dir='./example/pt_model_elec', num_labels=len(unique_labels))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model.load_state_dict(torch.load(checkpoint_dir), strict=False)

model.load_state_dict(torch.load(checkpoint_dir, map_location=torch.device('cpu')), strict=False)

model.to(device)



def predict():

    dataset = JsonDataset()
    dataloader = DataLoader(dataset, shuffle=False, batch_size=batch_size)
    
    result = []
    temp_i = 0
    with torch.no_grad():
        model.eval()
        t = tqdm(dataloader, ncols=80)
        for step, batch in enumerate(t):
            batch = {k: v.long().to(device) for k, v in batch.items()}
            res = model(  input_ids=batch['source'],
                            attention_mask=batch['attention_mask'],
                            labels=batch['label'])
            # predict = torch.argmax(res.logits, dim=-1).tolist()
            predict = torch.argmax(res.logits, dim=-1)
            predict[batch['attention_mask']==0] = -100
            predict = predict.tolist()
            
            for labels in predict:
                new_line = {}
                temp_id = id_list[temp_i]
                temp_sent = sent_list[temp_i]
                temp_i += 1
                new_line['sent_id'] = temp_id
                new_line['sent'] = temp_sent
                new_line['CourseGrainedErrorType'] = []
                new_line['FineGrainedErrorType'] = []
                prediction_label = [ids_to_labels[i] for i in labels if i != -100]
                for label in prediction_label:
                    if label == 'redu_sub':
                        if '成分赘余型错误' not in new_line['CourseGrainedErrorType']:
                            new_line['CourseGrainedErrorType'].append('成分赘余型错误')
                        if '主语多余' not in new_line['FineGrainedErrorType']:
                                new_line['FineGrainedErrorType'].append('主语多余')
                    elif label == 'redu_emp':
                        if '成分赘余型错误' not in new_line['CourseGrainedErrorType']:
                            new_line['CourseGrainedErrorType'].append('成分赘余型错误')
                        if '虚词多余' not in new_line['FineGrainedErrorType']:
                            new_line['FineGrainedErrorType'].append('虚词多余')
                    elif label == 'redu_other':
                        if '成分赘余型错误' not in new_line['CourseGrainedErrorType']:
                            new_line['CourseGrainedErrorType'].append('成分赘余型错误')
                        if '其他成分多余' not in new_line['FineGrainedErrorType']:
                            new_line['FineGrainedErrorType'].append('其他成分多余')
                    elif label == 'coll_order':
                        if '成分搭配不当型错误' not in new_line['CourseGrainedErrorType']:
                            new_line['CourseGrainedErrorType'].append('成分搭配不当型错误')
                        if '语序不当' not in new_line['FineGrainedErrorType']:
                            new_line['FineGrainedErrorType'].append('语序不当')
                    elif label == 'coll_vobj':
                        if '成分搭配不当型错误' not in new_line['CourseGrainedErrorType']:
                            new_line['CourseGrainedErrorType'].append('成分搭配不当型错误')
                        if '动宾搭配不当' not in new_line['FineGrainedErrorType']:
                            new_line['FineGrainedErrorType'].append('动宾搭配不当')
                    elif label == 'coll_other':
                        if '成分搭配不当型错误' not in new_line['CourseGrainedErrorType']:
                            new_line['CourseGrainedErrorType'].append('成分搭配不当型错误')
                        if '其他搭配不当' not in new_line['FineGrainedErrorType']:
                            new_line['FineGrainedErrorType'].append('其他搭配不当')
                    elif label == 'miss_sub':
                        if '成分残缺型错误' not in new_line['CourseGrainedErrorType']:
                            new_line['CourseGrainedErrorType'].append('成分残缺型错误')
                        if '主语不明' not in new_line['FineGrainedErrorType']:
                            new_line['FineGrainedErrorType'].append('主语不明')
                    elif label == 'miss_pre':
                        if '成分残缺型错误' not in new_line['CourseGrainedErrorType']:
                            new_line['CourseGrainedErrorType'].append('成分残缺型错误')
                        if '谓语残缺' not in new_line['FineGrainedErrorType']:
                            new_line['FineGrainedErrorType'].append('谓语残缺')
                    elif label == 'miss_obj':
                        if '成分残缺型错误' not in new_line['CourseGrainedErrorType']:
                            new_line['CourseGrainedErrorType'].append('成分残缺型错误')
                        if '宾语残缺' not in new_line['FineGrainedErrorType']:
                            new_line['FineGrainedErrorType'].append('宾语残缺')
                    elif label == 'miss_other':
                        if '成分残缺型错误' not in new_line['CourseGrainedErrorType']:
                            new_line['CourseGrainedErrorType'].append('成分残缺型错误')
                        if '其他成分残缺' not in new_line['FineGrainedErrorType']:
                            new_line['FineGrainedErrorType'].append('其他成分残缺')
                    elif label == 'char_append':
                        if '字符级错误' not in new_line['CourseGrainedErrorType']:
                            new_line['CourseGrainedErrorType'].append('字符级错误')
                        if '缺字漏字' not in new_line['FineGrainedErrorType']:
                            new_line['FineGrainedErrorType'].append('缺字漏字')
                    elif label == 'char_error':
                        if '字符级错误' not in new_line['CourseGrainedErrorType']:
                            new_line['CourseGrainedErrorType'].append('字符级错误')
                        if '错别字错误' not in new_line['FineGrainedErrorType']:
                            new_line['FineGrainedErrorType'].append('错别字错误')
                    elif label == 'char_punc_append':
                        if '字符级错误' not in new_line['CourseGrainedErrorType']:
                            new_line['CourseGrainedErrorType'].append('字符级错误')
                        if '缺少标点' not in new_line['FineGrainedErrorType']:
                            new_line['FineGrainedErrorType'].append('缺少标点')
                    elif label == 'char_punc_error':
                        if '字符级错误' not in new_line['CourseGrainedErrorType']:
                            new_line['CourseGrainedErrorType'].append('字符级错误')
                        if '错用标点' not in new_line['FineGrainedErrorType']:
                            new_line['FineGrainedErrorType'].append('错用标点')
                result.append(new_line)

    with open(result_path,'w',encoding='utf-8') as f:
        json.dump(result,f,ensure_ascii=False,indent=1)


pre_label = predict()
