from transformers.models.bert.modeling_bert import BertConfig, BertPreTrainedModel, BertModel, BertOnlyMLMHead, BertPreTrainedModel
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss

class BertForCSC(BertPreTrainedModel):
    def __init__(self, config):
        super(BertForCSC, self).__init__(config)
        self.bert = BertModel(config)
        # 24006
        config_seq = copy.copy(config)
        config_seq.vocab_size = 24006
        self.seq_cls = BertOnlyMLMHead(config_seq)
        # config_det = copy.copy(config)
        # config_det.vocab_size = 7
        # self.det_cls = BertOnlyMLMHead(config_det)
        self.init_weights()
    
    def forward(self, input_ids, attention_mask=None, token_labels=None, bi_labels=None):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]
        seq_scores = self.seq_cls(output)
        # det_scores = self.det_cls(output)

        if token_labels is not None :
            loss_fct = torch.nn.CrossEntropyLoss()
            seq_loss = loss_fct(seq_scores.view(-1, 24006), token_labels.view(-1))
            # det_loss = loss_fct(det_scores.view(-1, 7), bi_labels.view(-1))
            _, seq_select = torch.max(seq_scores, dim=-1)
            return seq_loss, seq_select
            # return seq_loss + det_loss, seq_select
        else:
            # seq_scores = seq_scores.softmax(-1)
            # seq_scores[...,2] = 0
            confi, seq_scores = torch.max(seq_scores, dim=-1)
            # _, det_scores = torch.max(det_scores, dim=-1)
            # return seq_scores, det_scores
            # return seq_scores, confi
            return seq_scores, seq_scores


if __name__ == '__main__':
    m = BertForCSC.from_pretrained('./pt_model/bert')
