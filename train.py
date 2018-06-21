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

def train(config):
    USE_CUDA = torch.cuda.is_available()
    DEVICE = torch.cuda.current_device()
    print('cuda states: {}'.format(USE_CUDA))

    SOURCE = Field(tokenize=str.split, use_vocab=True, init_token="<s>", eos_token="</s>", lower=True, 
                   include_lengths=True, batch_first=True)
    TARGET = Field(tokenize=str.split, use_vocab=True, init_token="<s>", eos_token="</s>", lower=True, 
                   batch_first=True)

    train_data, valid_data, test_data = \
        TabularDataset.splits(path=config.PATH, format='tsv', train=config.TRAIN_FILE, \
                              validation=config.VALID_FILE, test=config.TEST_FILE, \
                              fields=[('so', SOURCE), ('ta', TARGET)])

    SOURCE.build_vocab(train_data)
    TARGET.build_vocab(train_data)

    train_loader = BucketIterator(train_data, batch_size=config.BATCH, device=DEVICE,
                                  sort_key=lambda x: len(x.so), sort_within_batch=True, repeat=False)
    valid_loader = BucketIterator(valid_data, batch_size=config.BATCH, device=DEVICE,
                                  sort_key=lambda x: len(x.so), sort_within_batch=True, repeat=False)
    test_loader = BucketIterator(test_data, batch_size=config.BATCH, device=DEVICE,
                                  sort_key=lambda x: len(x.so), sort_within_batch=True, repeat=False)

    # parameters
    V_so = len(SOURCE.vocab)
    V_ta = len(TARGET.vocab)
    EARLY_STOPPING = False

    # build networks
    enc = Encoder(V_so, config.EMBED, config.HIDDEN, config.NUM_HIDDEN, bidrec=True)
    dec = Decoder(V_ta, config.EMBED, 2*config.HIDDEN, hidden_size2=config.HIDDEN2, \
                  sos_idx=SOURCE.vocab.stoi['<s>'], method=config.METHOD)
    if USE_CUDA:
        enc = enc.cuda()
        dec = dec.cuda()

    loss_function = nn.CrossEntropyLoss(ignore_index=TARGET.vocab.stoi['<pad>'])
    optimizer = optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=config.LR)
    scheduler = optim.lr_scheduler.MultiStepLR(gamma=0.1, milestones=[100], optimizer=optimizer)

    # train
    enc.train()
    dec.train()
    for step in range(config.STEP):
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
            valid_losses = evaluation(enc, dec, loss_function, valid_loader)
            msg = '[{}/{}] train_loss: {:.4f}, valid_loss: {:.4f}, lr: {}'.format(\
              step+1, config.STEP, np.mean(losses), np.mean(valid_losses), round(scheduler.get_lr()[0], 6))
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
    torch.save(enc.state_dict(), config.SAVE_ENC_PATH)
    torch.save(dec.state_dict(), config.SAVE_DEC_PATH)
    # test loss
    test_losses = evaluation(enc, dec, loss_function, test_loader)
    print('test_loss: {:.4f}'.format(test_losses))
    print('done!')
    
def evaluation(enc, dec, loss_function, loader):
    enc.eval()
    dec.eval()
    valid_losses = []

    for i, batch in enumerate(loader):
        inputs, lengths = batch.so
        targets = batch.ta

        output, hidden = enc(inputs, lengths.tolist())
        preds, _ = dec(hidden, output, lengths.tolist(), targets.size(1)) # max_len

        loss = loss_function(preds, targets.view(-1))
        valid_losses.append(loss.item())
            
    return valid_losses
