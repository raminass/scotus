from transformers import pipeline
from transformers import AutoTokenizer,BertModel
import torch.nn as nn
import torch
from transformers import DataCollatorWithPadding
import numpy as np

tokenizer = AutoTokenizer.from_pretrained('nlpaueb/legal-bert-small-uncased')
model = BertModel.from_pretrained('nlpaueb/legal-bert-small-uncased')
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

def embed_batch(batch):
  inputs = tokenizer(batch, return_tensors='pt', truncation=True, padding=True)
  with torch.no_grad():
    outputs = model(**inputs.to(device), output_hidden_states=True)
  last_hidden_states = outputs.last_hidden_state[:,0,:]
  return last_hidden_states.mean(dim=0)

def embed_case(df, case_name, category):
  lst = df[(df.case_name==case_name) & (df.category==category)]['text'].to_list()
  return embed_batch(lst).cpu().detach().numpy()