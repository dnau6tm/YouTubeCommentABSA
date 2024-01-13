from transformers import AutoModel, AutoTokenizer

import torch
from torch import nn


# Load pre-trained PhoBERT model and tokenizer
viso_model= AutoModel.from_pretrained('uitnlp/visobert')
viso_tokenizer = AutoTokenizer.from_pretrained('uitnlp/visobert')

class SentimentAnalysisModelViSo(nn.Module):
    def __init__(self, viso_model, num_labels):
        super(SentimentAnalysisModelViSo, self).__init__()
        self.viso = viso_model
        self.fc = nn.Linear(viso_model.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.viso(input_ids, attention_mask=attention_mask)
        logits = self.fc(outputs.pooler_output)
        return logits