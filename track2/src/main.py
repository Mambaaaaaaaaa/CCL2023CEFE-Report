import json
from tqdm import tqdm
from transformers import (
    BertTokenizer,
    set_seed,
    DataCollatorForSeq2Seq,
    StoppingCriteriaList, 
    MaxLengthCriteria,
    BeamSearchScorer
)
from model import BertForCSC
from torch.utils.data import DataLoader, Dataset
import datasets, argparse, logging, torch, os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW

def warmup_linear(x, warmup=0.002):
    if x < warmup:
        return x / warmup
    return max(1.0 - x, 0)

logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=47)
parser.add_argument('--train_json', type=str, default='')
parser.add_argument('--test_json', type=str, default='')
parser.add_argument('--data_cache', type=str, default='')
parser.add_argument('--pt_name', type=str, default='')
parser.add_argument('--pt_cache', type=str, default='')
parser.add_argument('--max_seq_len', type=int, default=128)
parser.add_argument('--cache_dir', type=str, default='')
parser.add_argument('--checkpoint_dir', type=str, default='')
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--num_train_epochs', type=int, default=10)
parser.add_argument('--do_train', action='store_true')
parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
parser.add_argument("--adam_epsilon", default=1e-8, type=float)
parser.add_argument('--warmup_rate', type=float, default=0.02)
parser.add_argument('--lr', type=float, required=True)

args = parser.parse_args()

def main():
    pass

def preprocess_function(examples):
    correct_map = {
        'correct': 0, 
        'char_error': 1,
        'char_append': 2,
        'char_delete': 3,
        'char_punc_error': 4,
        'char_punc_append': 5,
        'char_punc_delete': 6,
    }
    input_text = ['[CLS]']+list(examples['source'])
    bi_label = [correct_map[x] for x in examples['label']][:-1]
    # token_label = examples['token_labels'][:-1]
    token_label = examples['token_label'][:-1]
    
    assert len(input_text) == len(bi_label)
    assert len(input_text) == len(token_label)
    
    if len(input_text) >= args.max_seq_len-1:
        input_text = input_text[:args.max_seq_len-1]
        bi_label = bi_label[:args.max_seq_len-1]
        token_label = token_label[:args.max_seq_len-1]

    model_input = {}
    input_text = tokenizer.convert_tokens_to_ids(input_text+['[SEP]'])
    token_label = label_tokenizer.convert_tokens_to_ids(token_label+['$KEEP'])
    bi_label.append(0)
    mask = [1]*len(input_text)

    assert len(input_text) == len(token_label)
    
    while len(input_text) < args.max_seq_len:
        input_text.append(0)
        mask.append(0)
        token_label.append(-100)
        bi_label.append(-100)
    
    assert len(input_text) == args.max_seq_len
    assert len(token_label) == args.max_seq_len
    assert len(bi_label) == args.max_seq_len
    assert len(mask) == args.max_seq_len
    assert 'None' not in  token_label
    
    model_input['input_ids'] = input_text
    model_input['attention_mask'] = mask
    model_input['token_labels'] = token_label
    model_input['bi_labels'] = bi_label

    return model_input


def preprocess_function_test(examples):
    input_text = list(examples['sent'])
    # input_text = examples['source'][:-1]
    # bi_label = examples['bi_labels'][:-1]
    # token_label = examples['token_labels'][:-1]
    
    if len(input_text) >= args.max_seq_len-2:
        input_text = input_text[:args.max_seq_len-2]
        # input_text = input_text[:args.max_seq_len-1]
        # bi_label = bi_label[:args.max_seq_len-1]
        # token_label = token_label[:args.max_seq_len-1]

    model_input = {}
    input_text = tokenizer.convert_tokens_to_ids(['[CLS]']+input_text+['[SEP]'])
    # input_text = tokenizer.convert_tokens_to_ids(input_text+['[SEP]'])
    # token_label = label_tokenizer.convert_tokens_to_ids(token_label+['$KEEP'])
    # bi_label.append(0)
    mask = [1]*len(input_text)
    
    while len(input_text) < args.max_seq_len:
        input_text.append(0)
        mask.append(0)
        # token_label.append(-100)
        # bi_label.append(-100)
    
    assert len(input_text) == args.max_seq_len
    # assert len(token_label) == args.max_seq_len
    # assert len(bi_label) == args.max_seq_len
    assert len(mask) == args.max_seq_len
    # assert 'None' not in  token_label
    
    model_input['input_ids'] = input_text
    model_input['attention_mask'] = mask
    # model_input['token_labels'] = token_label
    # model_input['bi_labels'] = bi_label

    return model_input
    
def test():
    logger.info('test on epoch %d' % epoch_size)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=32)

    model.eval()
    input_ids_list = []
    attention_mask_list = []
    bi_predict_list = []
    token_predict_list = []
    bi_gold_list = []
    token_gold_list = []
    
    for step, batch in enumerate(tqdm(test_loader, desc='test')):
        # batch = [x.to(device) for x in batch]
        batch = {k: v.long().to(device) for k, v in batch.items()}

        bi_predict, token_predict = model(  input_ids=batch['input_ids'],
                        attention_mask=batch['attention_mask'])
        
        input_ids_list.extend(batch['input_ids'].tolist())
        attention_mask_list.extend(batch['attention_mask'].tolist())
        bi_predict_list.extend(bi_predict.tolist())
        token_predict_list.extend(token_predict.tolist())

        # bi_gold_list.extend(batch['bi_labels'].tolist())
        # token_gold_list.extend(batch['token_labels'].tolist())

    #bi
    p_bi = 0
    r_bi = 0
    tp_bi = 0
    p_to = 0
    r_to = 0
    tp_to = 0
    res = []
    for index in range(len(input_ids_list)):
        item = {}
        sent_len = sum(attention_mask_list[index])
        input_text = input_ids_list[index][:sent_len]
        # bi_pred = bi_predict_list[index][:sent_len]

        # token_gold = token_gold_list[index][:sent_len]
        # bi_gold = bi_gold_list[index][:sent_len]

        token_pred = token_predict_list[index][:sent_len]
        # print(''.join(tokenizer.decode(input_text)))

        item['input_text'] = tokenizer.decode(input_text)
        item['token_pred'] = label_tokenizer.decode(token_pred)
        res.append(item)

        # print(label_tokenizer.batch_decode(token_pred))
        # print(label_tokenizer.batch_decode(token_gold))
        # print(token_pred)
        # print(bi_pred)
        # print(bi_gold)
        # input()
    #     for j in range(sent_len):
    #         if bi_gold[j] != 0:
    #             r_bi += 1
    #             if bi_gold[j] == bi_pred[j]:
    #                 tp_bi += 1
    #         if bi_pred[j] != 0:
    #             p_bi += 1
    #         if token_gold[j] != 2:
    #             r_to += 1
    #             if token_gold[j] == token_pred[j]:
    #                 tp_to += 1
    #         if token_pred[j] != 2:
    #             p_to += 1
    # P_bi = tp_bi / (p_bi+1e-5)
    # R_bi = tp_bi / (r_bi+1e-5)
    # F_bi = P_bi*R_bi/(P_bi + R_bi)
    # P_to = tp_to / (p_to+1e-5)
    # R_to = tp_to / (r_to+1e-5)
    # F_to = 2*P_to*R_to/(P_to + R_to)
    json.dump(res, open('./data/output.json', 'w'), ensure_ascii=False, indent=2)
    # return (P_bi, R_bi, F_bi), (P_to, R_to, F_to)
        
    # input_ids = tokenizer.batch_decode(input_ids_list,  skip_special_tokens=True, clean_up_tokenization_spaces=True)
    # predict_list = tokenizer.batch_decode(predict_list,  skip_special_tokens=True, clean_up_tokenization_spaces=True)
    # res = []
    # for i, p in zip(input_ids, predict_list):
    #     item = {}
    #     item['sent'] = ''.join(i.split(' '))
    #     item['revisedSent'] = ''.join(p.split(' '))
    #     res.append(item)
    
    # json.dump(res, open('./predict.txt', 'w'), ensure_ascii=False, indent=2)
    
if __name__ == '__main__':
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )
    # device
    local_rank = int(os.environ["LOCAL_RANK"])
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend='nccl')
    
    set_seed(args.seed)
    
    tokenizer = BertTokenizer.from_pretrained('./pt_model/bert')
    label_tokenizer = BertTokenizer.from_pretrained('./pt_model/output', unk_token='$UNK')
    
    # data
    if args.do_train:
        raw_dataset = datasets.load_dataset('json', 
                                            data_files={'train': args.train_json},
                                            cache_dir=args.data_cache)

        column_names = raw_dataset["train"].column_names
        train_dataset = raw_dataset['train'].map(
            preprocess_function,
            remove_columns=[],
            batched=False,
            desc='running tokenizer on trainset'
        )
        train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'token_labels', 'bi_labels'])
        
    else:
        raw_dataset_test = datasets.load_dataset('json', 
                                            data_files={'test': args.test_json},
                                            cache_dir=args.data_cache)
        column_names_test = raw_dataset_test["test"].column_names
        test_dataset = raw_dataset_test['test'].map(
            preprocess_function_test,
            remove_columns=column_names_test,
            batched=False,
            desc='running tokenizer on testset'
        )
        test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
        # test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'token_labels', 'bi_labels'])
            
            
        test_loader = DataLoader(test_dataset, 
                                batch_size=args.batch_size,
                                shuffle=False
                                )

    # model
    model = BertForCSC.from_pretrained('./pt_model/bert', ignore_mismatched_sizes=True)
    model = model.to(local_rank)
    if args.checkpoint_dir != '':
        model.load_state_dict(torch.load(args.checkpoint_dir), strict=False)
        
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    
    epoch_size = 0
    if args.do_train:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
        train_loader = DataLoader(train_dataset, 
                                batch_size=args.batch_size,
                                sampler=train_sampler
                                )
        num_train_steps = int(len(train_dataset) / dist.get_world_size() / args.batch_size / args.gradient_accumulation_steps * args.num_train_epochs)
        param_optimizer = list(model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.lr, eps=args.adam_epsilon)
        
        global_step = 0
        best_score = 0
        
        for epoch_size in range(int(args.num_train_epochs)):
            model.train()
            logger.info(('train on epoch %d' % (epoch_size+1)))
            
            train_loader.sampler.set_epoch(epoch_size)
            
            t = tqdm(train_loader, desc=f'epoch{epoch_size}', ncols=80)
            
            for step, batch in enumerate(t):
                batch = {k: v.long().to(device) for k, v in batch.items()}

                loss, predict = model(  input_ids=batch['input_ids'],
                                attention_mask=batch['attention_mask'],
                                token_labels=batch['token_labels'],
                                bi_labels=batch['bi_labels'])

                predict[batch['token_labels']==-100] = -100
                
                correct = torch.sum(torch.all(predict == batch['token_labels'], dim=-1)) / (predict.size(0))
                
                loss.backward()

                # if dist.get_rank() == 0:
                #     writer.add_scalar('loss', loss.item(), global_step=global_step)
                    
                t.set_postfix_str('loss ={:^7.6f}&{:.4f}'.format(loss.item(), correct))
                # t.set_postfix_str('loss ={:^7.6f}&{:.4f}'.format(loss.item(), global_step/num_train_steps))
                t.update()
                
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    # modify learning rate with special warm up BERT uses
                    lr_this_step = args.lr * warmup_linear(global_step / num_train_steps, args.warmup_rate)
                    for param_group in optimizer.param_groups:
                        param_group['lr'] = lr_this_step
                    optimizer.step() 
                    optimizer.zero_grad()
                    global_step += 1
                
                if step % 1000 == 1 and dist.get_rank() == 0:
                    # res = test()
                    # print('bi P R F1 ', '  '.join([str(x) for x in res[0]]))
                    # print('token P R F1 ', '  '.join([str(x) for x in res[1]]))
    
                    # score = (res[0][2] + res[1][2] )/ 2
                    # if score >= best_score:
                    #     best_score = score
                    #     torch.save(model.module.state_dict(), args.cache_dir+'/checkpoint.tar.best')
                    # else:
                    torch.save(model.module.state_dict(), args.cache_dir+'/checkpoint.tar')
                    

            if dist.get_rank() == 0:
                torch.save(model.module.state_dict(), args.cache_dir+'/checkpoint.tar')
                
        
    else:
        if dist.get_rank() == 0:
            res = test()
            # print('bi P R F1 ', '  '.join([str(x) for x in res[0]]))
            # print('token P R F1 ', '  '.join([str(x) for x in res[1]]))
    
    
    
    

    
    
    