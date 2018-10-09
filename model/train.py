# coding utf-8
# import packages
import os
import torch
import torch.nn as nn
import torch.optim as optim
import spacy
from torchtext.data import Field, BucketIterator, TabularDataset
import torchtext.datasets as datasets

from decoder import Decoder
from encoder import Encoder

import numpy as np


def import_data(config, device):
    spacy_de = spacy.load('de')
    spacy_en = spacy.load('en')

    def tokenize_de(text):
        return [tok.text for tok in spacy_de.tokenizer(text)]

    def tokenize_en(text):
        return [tok.text for tok in spacy_en.tokenizer(text)]

    SRC = Field(tokenize=tokenize_de, 
                use_vocab=True, 
                lower=True, 
                include_lengths=True, 
                batch_first=True)
    TRG = Field(tokenize=tokenize_en, 
                use_vocab=True, 
                init_token='<s>', 
                eos_token='</s>', 
                lower=True, 
                batch_first=True)
    if config.DATATYPE == 'iwslt':
        train, valid, test = datasets.IWSLT.splits(exts=('.de', '.en'), fields=(SRC, TRG), root=config.ROOTPATH, filter_pred=lambda x: len(vars(x)['src']) <= config.MAX_LEN and len(vars(x)['trg']) <= config.MAX_LEN)
    elif config.DATATYPE == 'wmt':
        train, valid, test = datasets.WMT14.splits(exts=('.de', '.en'), fields=(SRC, TRG), root=os.path.join(config.ROOTPATH, 'wmt14'))

    SRC.build_vocab(train.src, min_freq=config.MIN_FREQ)
    TRG.build_vocab(train.trg, min_freq=config.MIN_FREQ)

    train_loader, valid_loader = BucketIterator.splits(datasets=(train, valid), batch_sizes=(config.BATCH, config.BATCH), sort_key=lambda x: len(x.src), sort_within_batch=True, repeat=False, device=device)
    return SRC, TRG, train, valid, train_loader, valid_loader 


def build_model(config, src_field, trg_field, device):
    enc = Encoder(V_e=len(src_field.vocab), 
                  m_e=config.EMBED, 
                  n_e=config.HIDDEN, 
                  num_layers=config.NUM_HIDDEN, 
                  bidrec=True, 
                  dropout_rate=config.DROPOUT_RATE, 
                  layernorm=config.LAYERNORM, 
                  device=device).to(device)
    dec = Decoder(V_d=len(trg_field.vocab), 
                  m_d=config.EMBED, 
                  n_d=2*config.HIDDEN,
                  num_layers=config.NUM_HIDDEN, 
                  hidden_size2=config.HIDDEN2,
                  sos_idx=trg_field.vocab.stoi['<s>'], 
                  method=config.METHOD, 
                  dropout_rate=config.DROPOUT_RATE,
                  layernorm=config.LAYERNORM, 
                  return_weight=config.RETURN_W, 
                  device=device).to(device)
    loss_function = nn.CrossEntropyLoss(ignore_index=trg_field.vocab.stoi['<pad>'])
    enc_optimizer = optim.Adam(enc.parameters(), 
                               lr=config.LR, 
                               weight_decay=config.LAMBDA)
    dec_optimizer = optim.Adam(dec.parameters(), 
                               lr=config.LR * config.DECLR, 
                               weight_decay=config.LAMBDA)
    enc_scheduler = optim.lr_scheduler.MultiStepLR(gamma=0.1,
                                                   milestones=[int(config.STEP / 4), 
                                                               int(config.STEP / 2),
                                                               int(3 * config.STEP / 4)],
                                                   optimizer=enc_optimizer)
    dec_scheduler = optim.lr_scheduler.MultiStepLR(gamma=0.1,
                                                   milestones=[int(config.STEP / 4), 
                                                               int(config.STEP / 2),
                                                               int(3 * config.STEP / 4)],
                                                   optimizer=dec_optimizer)
    return enc, dec, loss_function, enc_optimizer, dec_optimizer, enc_scheduler, dec_scheduler


def run_step(config, enc, dec, loader, loss_function, enc_optimizer, dec_optimizer):
    enc.train()
    dec.train()
    losses = []
    for i, batch in enumerate(loader):
        inputs, lengths = batch.src
        targets = batch.trg

        enc.zero_grad()
        dec.zero_grad()
        
        output, hidden = enc(inputs, lengths.tolist())
        
        if dec.return_weight:
            preds, weights = dec(hidden, output, lengths.tolist(), targets.size(1))  # max_len
        else:
            preds = dec(hidden, output, lengths.tolist(), targets.size(1))  # max_len
        loss = loss_function(preds, targets.view(-1))
        losses.append(loss.item())
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(enc.parameters(), 50.0)  # gradient clipping
        torch.nn.utils.clip_grad_norm_(dec.parameters(), 50.0)  # gradient clipping
        
        enc_optimizer.step()
        dec_optimizer.step()
        if i % config.PRINT_EVERY == 0:
            print(' > [{}/{}] train_loss {:.4f}'.format(i, len(loader), loss.item()))
    return np.mean(losses)


def validation(enc, dec, loader, loss_function):
    enc.eval()
    dec.eval()
    losses = []
    for i, batch in enumerate(loader):
        inputs, lengths = batch.src
        targets = batch.trg

        enc.zero_grad()
        dec.zero_grad()

        output, hidden = enc(inputs, lengths.tolist())
        preds, _ = dec(hidden, output, lengths.tolist(), targets.size(1))  # max_len

        loss = loss_function(preds, targets.view(-1))
        losses.append(loss.item())
        loss.backward()
    
    return np.mean(losses)


def train_model(config, enc, dec, loss_function, enc_optimizer, dec_optimizer, enc_scheduler, dec_scheduler, train_loader, valid_loader):
    print('--'*20)
    valid_losses=[9999]
    wait = 0
    for i, step in enumerate(range(config.STEP)):
        enc_scheduler.step()
        dec_scheduler.step()
        train_loss = run_step(config, enc, dec, train_loader, loss_function, enc_optimizer, dec_optimizer)
        if config.EMPTY_CUDA_MEMORY:
            torch.cuda.empty_cache()
        valid_loss = validation(enc, dec, valid_loader, loss_function)
        if config.EMPTY_CUDA_MEMORY:
            torch.cuda.empty_cache()
            
        valid_losses.append(valid_loss)

        print('[{}/{}] (train) loss {:.4f} | (valid) loss {:.4f} \n'.format(
                step+1, config.STEP, train_loss, valid_loss))

        # Save model
        if config.SAVE_MODEL:
            if config.SAVE_BEST:
                if valid_loss <= min(valid_losses):
                    torch.save(enc.state_dict(), config.SAVE_ENC_PATH)
                    torch.save(dec.state_dict(), config.SAVE_DEC_PATH)

                    print('****** model saved updated! ******')

            else:
                enc_model_path = config.SAVE_ENC_PATH + \
                    '{}_{:.4f}_{:.4f}'.format(step, train_loss, valid_loss)
                dec_model_path = config.SAVE_DEC_PATH + \
                    '{}_{:.4f}_{:.4f}'.format(step, train_loss, valid_loss)
                torch.save(enc.state_dict(), enc_model_path)
                torch.save(dec.state_dict(), dec_model_path)
                print('****** model saved updated! ******')
                
            if valid_loss > min(valid_losses):
                wait += 1
                if wait <= config.THRES:
                    print('****** early break!! ******')
                    break