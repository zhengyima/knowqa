from transformers import BertModel
import torch
import torch.nn as nn
import torch.nn.init as init

class BertForKnowledgeQA(nn.Module):
    def __init__(self, bert_model):
        super(BertForKnowledgeQA, self).__init__()
        self.bert_model = bert_model
    
    def forward(self, batch_data):
        """
        Args:
            input_ids ([type]): [description]
            attention_mask ([type]): [description]
            token_type_ids ([type]): [description]
        """
        input_ids = batch_data["input_ids"]
        labels = batch_data["labels"]
        attention_mask = batch_data['attention_mask']
        bert_inputs = {'input_ids': input_ids, 'labels': labels, 'attention_mask':attention_mask}
        model_output = self.bert_model(**bert_inputs)

        return model_output