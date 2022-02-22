import torch
from transformers import PretrainedConfig, AutoModelForCausalLM
import json

config_path = 'downloaded/comet-distill'
model = AutoModelForCausalLM.from_pretrained(config_path)
model.eval()

from transformers import AutoTokenizer, GPT2Tokenizer

tokenizer_path = 'downloaded/comet-distill-tokenizer'
#tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
tokenizer = GPT2Tokenizer.from_pretrained(tokenizer_path)

def load_atomic_light(datapath):
  rels = ['result', 'want', 'need']
  rel_tokens = {'result': 'xEffect', 'want': 'xWant', 'need': 'xNeed'}
  with open(datapath, 'r') as f:
    lines = f.readlines()
  print("EXTRACTED LINES")
  data = []
  for line in lines:
    #Still need to separate out relation
    line = line.split('\t')
    line = line[:2]
    line = [item.split(':')[1] for item in line]
    rel_token = None
    for rel in rels:
      if rel in line[0]:
        rel_token = rel_tokens[rel]
        break
    line.insert(1, rel_token)
    line[0] = "<head> " + line[0] + "</head>"
    line[1] = "<relation> " + line[1] + "</relation>"
    line.insert(2, "[GEN]")
    data.append(line)
  return data

def datapoint_to_text_input(datapoint):
  return datapoint[0]+datapoint[1]+datapoint[2]+datapoint[3]

import pandas as pd
import json
datafile = 'atomic_tuned/bert/atomic_sample_train.txt'
data = load_atomic_light(datafile)
print(len(data))
print(data[:5])
max_len = 0
for datapoint in data:
  text = datapoint[0]+datapoint[1]+datapoint[2]+datapoint[3]
  tokenized = tokenizer(text)["input_ids"]
  max_len = max(max_len, len(tokenized))
print(max_len)

datapoint = data[0]
text = datapoint_to_text_input(datapoint)
print(text)
tokenized = tokenizer(text)
print(tokenized)

import pandas as pd
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm, trange
import torch.nn.functional as F
import csv

import os
import time
import datetime

import pandas as pd
import numpy as np
import random


import torch
from torch.utils.data import Dataset, DataLoader, random_split, RandomSampler, SequentialSampler
torch.manual_seed(42)

from transformers import GPT2LMHeadModel,  GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
from transformers import AdamW, get_linear_schedule_with_warmup

class AtomicLight(Dataset):
    def __init__(self, data, tokenizer, max_length=100):

        self.tokenizer = tokenizer
        self.input_ids = []
        self.attn_masks = []
        self.labels = []

        for datapoint in data:
          input = datapoint_to_text_input(datapoint)
          encodings_dict = tokenizer(input, max_length=max_length, padding="max_length")
          datapoint_input_ids = torch.tensor(encodings_dict['input_ids'])
          self.input_ids.append(datapoint_input_ids)
          self.attn_masks.append(torch.tensor(encodings_dict['attention_mask']))
          label_mask = torch.zeros_like(datapoint_input_ids,dtype=torch.long)-100
          #50269 is [GEN] token
          GEN_TOKEN = 50269
          gen_index = (datapoint_input_ids == GEN_TOKEN).nonzero(as_tuple=True)[0]
          label_mask[(gen_index+1):] = datapoint_input_ids[(gen_index+1):]
          self.labels.append(label_mask)
        self.length = len(data)

    def __len__(self):
        return self.length

    def __getitem__(self, item):
        return self.input_ids[item], self.attn_masks[item], self.labels[item]

dataset = AtomicLight(data, tokenizer)
train_size = int(.9*len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
batch_size = 8
train_dataloader = DataLoader(
    train_dataset,
    sampler=RandomSampler(train_dataset),
    batch_size=batch_size
)

validation_dataloader = DataLoader(
    val_dataset,
    sampler=SequentialSampler(val_dataset),
    batch_size=batch_size
)

device = torch.device('cuda')
model.to(device)

# some parameters I cooked up that work reasonably well

epochs = 5
learning_rate = 5e-4
warmup_steps = 1e2
epsilon = 1e-8

# this produces sample output every 100 steps
sample_every = 100

# Note: AdamW is a class from the huggingface library (as opposed to pytorch)
optimizer = AdamW(model.parameters(),
                  lr = learning_rate,
                  eps = epsilon
                )

# Total number of training steps is [number of batches] x [number of epochs].
# (Note that this is not the same as the number of training samples).
total_steps = len(train_dataloader) * epochs
steps_per_epoch = len(train_dataloader)

# Create the learning rate scheduler.
# This changes the learning rate as the training loop progresses
scheduler = get_linear_schedule_with_warmup(optimizer,
                                            num_warmup_steps = warmup_steps,
                                            num_training_steps = total_steps)


#wandb.init(project="distill-atomic-light")

for epoch_i in range(0, epochs):

    # ========================================
    #               Training
    # ========================================

    print("")
    print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
    print('Training...')

    total_train_loss = 0

    model.train()

    for step, batch in tqdm(enumerate(train_dataloader)):

        b_input_ids = batch[0].to(device)
        b_labels = batch[2].to(device)
        b_masks = batch[1].to(device)

        model.zero_grad()

        outputs = model(  b_input_ids,
                          labels=b_labels,
                          attention_mask = b_masks,
                          token_type_ids=None
                        )

        loss = outputs[0]

        batch_loss = loss.item()
        if step % 100 == 0:
          print(f'Epoch {epoch_i} out of {epochs}')
          print(f'iteration {step} out of {steps_per_epoch}')
          print('loss: ', batch_loss)
        total_train_loss += batch_loss

        # Get sample every x batches.

        loss.backward()

        optimizer.step()

        scheduler.step()

    # Calculate the average loss over all of the batches.
    avg_train_loss = total_train_loss / len(train_dataloader)


    print("")
    print("  Average training loss: {0:.2f}".format(avg_train_loss))

    # ========================================
    #               Validation
    # ========================================

    print("")
    print("Running Validation...")

    model.eval()

    total_eval_loss = 0
    nb_eval_steps = 0

    # Evaluate data for one epoch
    for batch in validation_dataloader:

        b_input_ids = batch[0].to(device)
        b_labels = batch[0].to(device)
        b_masks = batch[1].to(device)

        with torch.no_grad():

            outputs  = model(b_input_ids,
#                            token_type_ids=None,
                             attention_mask = b_masks,
                            labels=b_labels)

            loss = outputs[0]

        batch_loss = loss.item()
        print('val loss: ', batch_loss)
        total_eval_loss += batch_loss

    avg_val_loss = total_eval_loss / len(validation_dataloader)

    print("  Validation Loss: {0:.2f}".format(avg_val_loss))

model.save_pretrained('.')


print("")
print("Training complete!")