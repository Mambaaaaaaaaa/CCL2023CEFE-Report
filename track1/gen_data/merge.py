import json 

path_1 = 'gen_'
path_2 = '_label.json'

result = {}
result['source'] = []
result['target'] = []
result['label'] = []


val = {}
val['source'] = []
val['target'] = []
val['label'] = []

test = {}
test['source'] = []
test['target'] = []
test['label'] = []

cls = '[CLS]'
sep = '[SEP]'

for i in range(1,5):
    new_path = path_1 + str(i) + path_2
    with open(new_path, 'r',encoding='utf-8') as f:
        string = f.read()
        raw_data = json.loads(string)
        m = 0
        for sub in raw_data:
            source = sub['source'][5:-5]
            source = cls + source + sep
            target = sub['target'][5:-5]
            target = cls + target + sep
            label = sub['label']
            if m < len(raw_data) * 0.9:
                result['source'].append(source)
                result['target'].append(target)
                result['label'].append(label)
            else :
                val['source'].append(source)
                val['target'].append(target)
                val['label'].append(label)
            m += 1



with open("./data/train_char.json",'w',encoding='utf-8') as f:
        json.dump(result,f,ensure_ascii=False,indent=1)

with open("./data/val_char.json",'w',encoding='utf-8') as f:
        json.dump(val,f,ensure_ascii=False,indent=1)


# with open("test_char.json",'w',encoding='utf-8') as f:
#         json.dump(test,f,ensure_ascii=False,indent=1)
