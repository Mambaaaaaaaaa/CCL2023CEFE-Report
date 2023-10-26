import json
from tqdm import tqdm
from transformers import (
    BertTokenizer,
    BartForConditionalGeneration,
    set_seed,
    DataCollatorForSeq2Seq,
    StoppingCriteriaList, 
    MaxLengthCriteria,
    BeamSearchScorer
)
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
    inputs = examples['sent']
    targets = examples['revisedSent']
    # inputs = examples['new_source']
    # targets = examples['new_target']

    model_inputs = tokenizer(inputs, max_length=args.max_seq_len, padding='max_length', truncation=True)
    labels = tokenizer(text_target=targets, max_length=args.max_seq_len, padding='max_length', truncation=True)
    labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def preprocess_function_test(examples):
    inputs = examples['sent']
    # targets = examples['revisedSent']

    model_inputs = tokenizer(inputs, max_length=args.max_seq_len, padding='max_length', truncation=True)
    # labels = tokenizer(text_target=targets, max_length=args.max_seq_len, padding='max_length', truncation=True)
    # labels["input_ids"] = [
    #             [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
    #         ]
    # model_inputs["labels"] = labels["input_ids"]
    return model_inputs
    
def test():
    logger.info('test on epoch %d' % epoch_size)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=32)

    model.eval()
    input_ids_list = []
    predict_list = []
    
    for step, batch in enumerate(tqdm(test_loader, desc='test')):
        # batch = [x.to(device) for x in batch]
        batch = {k: v.long().to(device) for k, v in batch.items()}
        # logits_processor = LogitsProcessorList([ MinLengthLogitsProcessor(batch_len, eos_token_id=model.config.eos_token_id),])
        stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=128),])
        
        decoder_input_ids = torch.ones((batch['input_ids'].size(0), 1), dtype=torch.long, device=batch['input_ids'].device) * model.module.config.decoder_start_token_id
        predict = model.module.greedy_search(input_ids=decoder_input_ids,
                                    encoder_outputs=model.module.get_encoder()( input_ids=batch['input_ids'], attention_mask=batch['attention_mask']),
                                    attention_mask=batch['attention_mask'],
                                    stopping_criteria=stopping_criteria)

        # beam_scorer = BeamSearchScorer(
        #     batch_size=decoder_input_ids.size(0),
        #     num_beams=4,
        #     device=batch['input_ids'].device,
        # )
        # predict = model.module.beam_search(input_ids=decoder_input_ids.repeat_interleave(4, dim=0),
        #                             beam_scorer=beam_scorer,
        #                             encoder_outputs=model.module.get_encoder()( input_ids=batch['input_ids'].repeat_interleave(4, dim=0), attention_mask=batch['attention_mask'].repeat_interleave(4, dim=0)),
        #                             attention_mask=batch['attention_mask'].repeat_interleave(4, dim=0),
        #                             # decoder_attention_mask=decoder_attention_mask,
        #                             # logits_processor=logits_processor,
        #                             stopping_criteria=stopping_criteria)
        input_ids_list.extend(batch['input_ids'].tolist())
        predict_list.extend(predict.tolist())

    input_ids = tokenizer.batch_decode(input_ids_list,  skip_special_tokens=True, clean_up_tokenization_spaces=True)
    predict_list = tokenizer.batch_decode(predict_list,  skip_special_tokens=True, clean_up_tokenization_spaces=True)
    res = []
    for i, p in zip(input_ids, predict_list):
        item = {}
        item['sent'] = ''.join(i.split(' '))
        item['revisedSent'] = ''.join(p.split(' '))
        res.append(item)
    
    json.dump(res, open('./predict.txt', 'w'), ensure_ascii=False, indent=2)
    
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
    
    tokenizer = BertTokenizer.from_pretrained(args.pt_name)
    # token_list = '<char_append> </char_append> <char_error> </char_error> <char_delete> </char_delete> <char_punc_append> </char_punc_append> <char_punc_error> </char_punc_error> <char_punc_delete> </char_punc_delete> <miss_sub> </miss_sub> <miss_pre> </miss_pre> <miss_obj> </miss_obj> <miss_other> </miss_other> <redu_sub> </redu_sub> <redu_emp> </redu_emp> <redu_other> </redu_other> <coll_order> </coll_order> <coll_vobj> </coll_vobj> <coll_other> </coll_other>'.split(' ')
    # tokenizer.add_tokens(token_list, special_tokens=True)
    
    # data
    if args.do_train:
        raw_dataset = datasets.load_dataset('json', 
                                            data_files={'train': args.train_json},
                                            cache_dir=args.data_cache)

        column_names = raw_dataset["train"].column_names
        train_dataset = raw_dataset['train'].map(
            preprocess_function,
            remove_columns=column_names,
            batched=True,
            desc='running tokenizer on trainset'
        )
        train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
        
    else:
        raw_dataset = datasets.load_dataset('json', 
                                            data_files={'test': args.test_json},
                                            cache_dir=args.data_cache)
        column_names_test = raw_dataset["test"].column_names
        test_dataset = raw_dataset['test'].map(
            preprocess_function_test,
            remove_columns=column_names_test,
            batched=True,
            desc='running tokenizer on testset'
        )
        test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
        
        
        test_loader = DataLoader(test_dataset, 
                              batch_size=args.batch_size,
                              shuffle=False
                              )

    # model
    # model = BartForConditionalGeneration.from_pretrained(args.pt_name, cache_dir=args.pt_cache)
    model = BartForConditionalGeneration.from_pretrained(args.pt_name)
    model = model.to(local_rank)
    if args.checkpoint_dir != '':
        model.load_state_dict(torch.load(args.checkpoint_dir))
        
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    
    
    
    # test_dataset = raw_dataset['test'].map(
    #     preprocess_function_test,
    #     remove_columns=column_names_test,
    #     batched=True,
    #     desc='running tokenizer on testset'
    # )
    # test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])
    
    # label_pad_token_id = -100
    # data_collator = DataCollatorForSeq2Seq(
    #         tokenizer,
    #         model=model,
    #         label_pad_token_id=label_pad_token_id,
    #         pad_to_multiple_of=None,
    #     )

    
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
        
        for epoch_size in range(int(args.num_train_epochs)):
            model.train()
            logger.info(('train on epoch %d' % (epoch_size+1)))
            
            train_loader.sampler.set_epoch(epoch_size)
            
            t = tqdm(train_loader, desc=f'epoch{epoch_size}', ncols=80)
            
            for step, batch in enumerate(t):
                batch = {k: v.long().to(device) for k, v in batch.items()}

                output = model(  input_ids=batch['input_ids'],
                                attention_mask=batch['attention_mask'],
                                labels=batch['labels'])
                loss = output.loss
                predict = torch.argmax(output.logits, dim=-1)
                
                predict[batch['labels']==-100] = -100
                
                correct = torch.sum(torch.all(predict == batch['labels'], dim=-1)) / (predict.size(0))
                
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
                    # cur_f1 = test_correct()
                    # cur_f1_r = test_correct_r()
                    # if (cur_f1[1][2] + cur_f1_r[1][2])/2 > best_f1:
                    # if ( cur_f1[1][2]) > best_f1:
                    #     best_f1 = cur_f1[1][2]
                    #     # best_f1 = (cur_f1[1][2] + cur_f1_r[1][2])/2
                    #     logger.info(('save_model', str(best_f1)))
                    #     torch.save(model.module.state_dict(), args.cache_dir+'/checkpoint.tar.best')
                    # else:
                    torch.save(model.module.state_dict(), args.cache_dir+'/checkpoint.tar')
                    

            if dist.get_rank() == 0:
            #     # torch.save(model.module.state_dict(), args.cache_dir+'/checkpoint.tar')
            #     # test_correct()
            #     # cur_f1 = test_correct()
            #     # cur_f1_r = test_correct_r()
            #     if ( cur_f1[1][2]) > best_f1:
            #     # if (cur_f1[1][2] + cur_f1_r[1][2])/2 > best_f1:
            #         # best_f1 = cur_f1[1][2]
            #         best_f1 = cur_f1[1][2]
            #         # best_f1 = (cur_f1[1][2] + cur_f1_r[1][2])/2
            #         logger.info(('save_model', str(best_f1)))
            #         # torch.save(model.module.state_dict(), args.cache_dir+'/checkpoint.tar')
            #         torch.save(model.module.state_dict(), args.cache_dir+'/checkpoint.tar.best')
            #     else:
                torch.save(model.module.state_dict(), args.cache_dir+'/checkpoint.tar')
                
        
    else:
        if dist.get_rank() == 0:
            test()
    
    
    
    

    
    
    