import json
import datasets
from transformers import BertTokenizer

def preprocess_function(examples):
    correct_map = {
        'correct': 0, 
        'char_error': 1,
        'char_append': 2,
        'char_punc_error': 3,
        'char_punc_append': 4,
    }
    print(examples)
    input_text = ['[CLS]']+list(examples['source'][5:-5])
    bi_label = [correct_map[x] for x in examples['label']][:-1]
    token_label = ['$K'] * len(input_text)
    assert len(input_text) == len(bi_label)
    target = ['[CLS]']+list(examples['target'][5:-5])
    cur_index = 0
    for i in range(len(input_text)):
        if bi_label[i] == 0:
            token_label[i] = '$KEEP'
        elif bi_label[i] == 1:
            token_label[i] = '$REPLACE_'+target[cur_index]
        elif bi_label[i] == 2:
            cur_index += 1
            token_label[i] = '$APPEND_'+target[cur_index]
        elif bi_label[i] == 3:
            token_label[i] = '$REPLACE_'+target[cur_index]
        elif bi_label[i] == 4:
            cur_index += 1
            token_label[i] = '$APPEND_'+target[cur_index]
        cur_index += 1
    
    if len(input_text) >= max_seq_len-1:
        input_text = input_text[:max_seq_len-1]
        bi_label = bi_label[:max_seq_len-1]
        token_label = token_label[:max_seq_len-1]

    model_input = {}
    input_text = tokenizer.convert_tokens_to_ids(input_text+['[SEP]'])
    token_label = label_tokenizer.convert_tokens_to_ids(token_label+['$KEEP'])
    bi_label.append(0)
    mask = [1]*len(input_text)
    while len(input_text) < max_seq_len:
        input_text.append(0)
        mask.append(0)
        token_label.append(-100)
        bi_label.append(-100)

    assert len(input_text) == max_seq_len
    assert len(token_label) == max_seq_len
    assert len(bi_label) == max_seq_len
    assert len(mask) == max_seq_len
    
    model_input['input_ids'] = input_text
    model_input['attention_mask'] = mask
    model_input['token_labels'] = token_label
    model_input['bi_labels'] = input_text
    
    return model_input

if __name__ == '__main__':
    # max_seq_len = 128
    # t_d = datasets.load_dataset('json', data_files='./pseudo_data/new_char_0601.json')['train']
    # tokenizer = BertTokenizer.from_pretrained('./pt_model/bert')
    # label_tokenizer = BertTokenizer.from_pretrained('./pt_model/output')
    # t_d.map(preprocess_function,
            # remove_columns=[])
    # print(t_d)
    
    raw_dataset = datasets.load_dataset('json', data_files={'train': './pseudo_data/new_char_0601.json'})