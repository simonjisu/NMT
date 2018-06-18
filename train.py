# coding utf-8
# import packages
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchtext.data import Field, Iterator, BucketIterator, TabularDataset
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from decoder import Decoder
from encoder import Encoder
from attention import Attention

import numpy as np
USE_CUDA = torch.cuda.is_available()
DEVICE = torch.cuda.current_device()

# data loader settings
train_file = 'eng-fra-filtered.train'
valid_file = 'eng-fra-filtered.valid'
test_file = 'eng-fra-filtered.test'
BATCH_SIZE = 64

SOURCE = Field(tokenize=str.split, use_vocab=True, init_token="<s>", eos_token="</s>", lower=True, 
               include_lengths=True, batch_first=True)
TARGET = Field(tokenize=str.split, use_vocab=True, init_token="<s>", eos_token="</s>", lower=True, 
               batch_first=True)

train_data, valid_data, test_data = \
    TabularDataset.splits(path='data/en_fa/', format='tsv', train=train_file, validation=valid_file, test=test_file, fields=[('so', SOURCE), ('ta', TARGET)])

SOURCE.build_vocab(train_data)
TARGET.build_vocab(train_data)

train_loader = BucketIterator(train_data, batch_size=BATCH_SIZE, device=DEVICE,
                              sort_key=lambda x: len(x.so), sort_within_batch=True, repeat=False)
valid_loader = BucketIterator(valid_data, batch_size=BATCH_SIZE, device=DEVICE,
                              sort_key=lambda x: len(x.so), sort_within_batch=True, repeat=False)
test_loader = BucketIterator(test_data, batch_size=BATCH_SIZE, device=DEVICE,
                              sort_key=lambda x: len(x.so), sort_within_batch=True, repeat=False)

# parameters
V_so = len(SOURCE.vocab)
V_ta = len(TARGET.vocab)
HIDDEN = 1000
EMBED = 500
STEP = 200
LR = 0.001
NUM_LAYERS = 1
BATCH_SIZE = BATCH_SIZE
EARLY_STOPPING = False

# build networks
enc = Encoder(V_so, EMBED, HIDDEN, NUM_LAYERS, bidrec=True)
dec = Decoder(V_ta, EMBED, 2*HIDDEN, sos_idx=SOURCE.vocab.stoi['<s>'], method='general')
if USE_CUDA:
    enc = enc.cuda()
    dec = dec.cuda()

loss_function = nn.CrossEntropyLoss(ignore_index=TARGET.vocab.stoi['<pad>'])
optimizer = optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=LR)
scheduler = optim.lr_scheduler.MultiStepLR(gamma=0.1, milestones=[100], optimizer=optimizer)

# train
enc.train()
dec.train()
for step in range(STEP):
    losses=[]
    scheduler.step()
    if EARLY_STOPPING:
        break
    for i, batch in enumerate(train_loader):
        inputs, lengths = batch.so
        targets = batch.ta
        
        enc.zero_grad()
        dec.zero_grad()
        
        output, hidden = enc(inputs, lengths.tolist())
        preds, _ = dec(hidden, output, lengths.tolist(), targets.size(1)) # max_len
        
        loss = loss_function(preds, targets.view(-1))
        losses.append(loss.item())
        
        loss.backward()
        optimizer.step()
    
    if step % 10 == 0:
        enc.eval()
        dec.eval()
        valid_losses = []
        for i, batch in enumerate(valid_loader):
            inputs, lengths = batch.so
            targets = batch.ta
            
            output, hidden = enc(inputs, lengths.tolist())
            preds, _ = dec(hidden, output, lengths.tolist(), targets.size(1)) # max_len

            loss = loss_function(preds, targets.view(-1))
            valid_losses.append(loss.item())

        msg = '[{}/{}] train_loss: {:.4f}, valid_loss: {:.4f}, lr: {}'.format(\
          step+1, STEP, np.mean(losses), np.mean(valid_losses), round(scheduler.get_lr()[0], 6))
        print(msg)
        
        if np.mean(valid_losses) < 0.1:
            EARLY_STOPPING = True
            print("Early Stopping!")
            break
        valid_losses = []
        losses = []
        enc.train()
        dec.train()

# save model
torch.save(enc.state_dict(), './data/model/fra_eng.enc')
torch.save(dec.state_dict(), './data/model/fra_eng.dec')