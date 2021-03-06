# coding utf-8
# import packages
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import spacy
from torchtext.data import Field, BucketIterator, TabularDataset
import torchtext.datasets as datasets
from konlpy.tag import Mecab

from decoder import Decoder
from encoder import Encoder

import numpy as np
# import wandb

def import_data(config, device, is_test=False):
    spacy_de = spacy.load('de')
    spacy_en = spacy.load('en')
    tagger = Mecab()

    def tokenize_de(text):
        return [tok.text for tok in spacy_de.tokenizer(text)]

    def tokenize_en(text):
        return [tok.text for tok in spacy_en.tokenizer(text)]
    
    if config.DATATYPE == 'eng-kor':
        src_tokenizer = tokenize_en
        trg_tokenizer = tagger.morphs
    else:
        src_tokenizer = tokenize_de
        trg_tokenizer = tokenize_en
    
    SRC = Field(tokenize=src_tokenizer, 
                use_vocab=True, 
                lower=True, 
                include_lengths=True, 
                batch_first=True)
    TRG = Field(tokenize=trg_tokenizer, 
                use_vocab=True,
                init_token='<s>',
                eos_token='</s>', 
                lower=True, 
                batch_first=True)
    if config.DATATYPE == 'iwslt':
        train, valid, test = datasets.IWSLT.splits(
            exts=('.de', '.en'), 
            fields=(SRC, TRG), 
            root=config.ROOTPATH, 
            filter_pred=lambda x: len(x.src) <= config.MAX_LEN and len(x.trg) <= config.MAX_LEN)
    elif config.DATATYPE == 'wmt':
        train, valid, test = datasets.WMT14.splits(
            exts=('.de', '.en'), 
            fields=(SRC, TRG), 
            root=config.ROOTPATH, 
            filter_pred=lambda x: len(x.src) <= config.MAX_LEN and len(x.trg) <= config.MAX_LEN)
    elif config.DATATYPE == 'eng-kor':
        train, valid, test = TabularDataset.splits(
            path=os.path.join(config.ROOTPATH, 'en_kr'),
            format='tsv',
            fields=[('src', SRC), ('trg', TRG)],
            train='eng-kor.train', 
            validation='eng-kor.valid', 
            test='eng-kor.test')

    SRC.build_vocab(train.src, min_freq=config.MIN_FREQ)
    TRG.build_vocab(train.trg, min_freq=config.MIN_FREQ)
    print("Source Language: {} words, Target Language: {} words".format(len(SRC.vocab), len(TRG.vocab)))
    print("Training Examples: {}, Validation Examples: {}".format(len(train), len(valid)))
    if is_test:
        train_loader, valid_loader, test_loader = BucketIterator.splits(
            datasets=(train, valid, test), 
            batch_sizes=(config.BATCH, config.BATCH, config.BATCH), 
            sort_key=lambda x: len(x.src), 
            sort_within_batch=True, 
            repeat=False, 
            device=device)
        return SRC, TRG, train, valid, test, train_loader, valid_loader, test_loader
    else:
        train_loader, valid_loader = BucketIterator.splits(
            datasets=(train, valid), 
            batch_sizes=(config.BATCH, config.BATCH), 
            sort_key=lambda x: len(x.src), 
            sort_within_batch=True, 
            repeat=False, 
            device=device)
        
        return SRC, TRG, train, valid, train_loader, valid_loader 


def build_model(config, src_field, trg_field, device):
    """
    enc, dec, loss_function, enc_optimizer, dec_optimizer, enc_scheduler, dec_scheduler
    """
    enc = Encoder(len(src_field.vocab), 
                  config.EMBED, 
                  config.HIDDEN, 
                  config.ENC_N_LAYER, 
                  layernorm=config.L_NORM, 
                  bidirec=True).to(device)

    dec = Decoder(len(trg_field.vocab), 
                  config.EMBED, 
                  enc.n_direction*config.HIDDEN, 
                  config.DEC_N_LAYER, 
                  drop_rate=config.DROP_RATE, 
                  method=config.METHOD, 
                  layernorm=config.L_NORM, 
                  sos_idx=trg_field.vocab.stoi['<s>'],
                  teacher_force=config.TF, 
                  return_w=config.RETURN_W, 
                  device=device).to(device)
    
    loss_function = nn.CrossEntropyLoss(ignore_index=trg_field.vocab.stoi['<pad>'])
    if config.OPTIM.lower() == 'adam':
        enc_optimizer = optim.Adam(enc.parameters(), 
                                   lr=config.LR, 
                                   weight_decay=config.LAMBDA)
        dec_optimizer = optim.Adam(dec.parameters(), 
                                   lr=config.LR * config.DECLR, 
                                   weight_decay=config.LAMBDA)
    elif config.OPTIM.lower() == 'adelta':
        enc_optimizer = optim.Adadelta(enc.parameters(),
                                       weight_decay=config.LAMBDA)
        dec_optimizer = optim.Adadelta(dec.parameters(),
                                       weight_decay=config.LAMBDA)
    elif config.OPTIM.lower() == 'sgd':
        enc_optimizer = optim.SGD(enc.parameters(),
                                   lr=config.LR,
                                   weight_decay=config.LAMBDA)
        dec_optimizer = optim.SGD(dec.parameters(),
                                   lr=config.LR * config.DECLR,
                                   weight_decay=config.LAMBDA)
    enc_scheduler = optim.lr_scheduler.MultiStepLR(gamma=0.1,
                                                   milestones=[int(config.STEP / 4),
                                                               int(2 * config.STEP / 3)],
                                                   optimizer=enc_optimizer)
    dec_scheduler = optim.lr_scheduler.MultiStepLR(gamma=0.1,
                                                   milestones=[int(config.STEP / 4), 
                                                               int(2 * config.STEP / 3)],
                                                   optimizer=dec_optimizer)
    print("Building Model ...")
    return enc, dec, loss_function, enc_optimizer, dec_optimizer, enc_scheduler, dec_scheduler


def run_step(config, enc, dec, loader, loss_function, enc_optimizer, dec_optimizer):
    enc.train()
    dec.train()
    losses = []
    pee = len(loader) // config.PRINT_EVERY
    for i, batch in enumerate(loader):
        inputs, lengths = batch.src
        targets = batch.trg

        enc.zero_grad()
        dec.zero_grad()
        
        enc_output, enc_hidden = enc(inputs, lengths.tolist())
        outputs = dec(enc_hidden, enc_output, lengths.tolist(), 
                      targets.size(1), targets, is_eval=False)
        loss = loss_function(outputs, targets[:, 1:].contiguous().view(-1))
        losses.append(loss.item())
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(enc.parameters(), 50.0)  # gradient clipping
        torch.nn.utils.clip_grad_norm_(dec.parameters(), 50.0)  # gradient clipping
        
        enc_optimizer.step()
        dec_optimizer.step()
        if i % pee == 0:
            print(' > [{}/{}] train_loss {:.4f}'.format(i, len(loader), loss.item()))
        # wandb test
        # wandb.log({"train_loss": np.mean(losses)})
        # wandb test
    return np.mean(losses)


def validation(config, enc, dec, loader, loss_function):
    enc.eval()
    dec.eval()
    losses = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            inputs, lengths = batch.src
            targets = batch.trg

            output, hidden = enc(inputs, lengths.tolist())
            if dec.return_w:
                preds, attns = dec(hidden, output, lengths.tolist(), 
                                   targets.size(1), targets, is_eval=config.TF) 
            else:
                preds = dec(hidden, output, lengths.tolist(), targets.size(1), targets, is_eval=config.TF)
            loss = loss_function(preds, targets[:, 1:].contiguous().view(-1))
            losses.append(loss.item())
    # wandb test
    # wandb.log({"valid_loss": np.mean(losses)})
    # wandb test
    return np.mean(losses)


def train_model(config, enc, dec, loss_function, enc_optimizer, dec_optimizer, enc_scheduler, dec_scheduler, train_loader, valid_loader):
    
    valid_losses = [validation(config, enc, dec, valid_loader, loss_function)] if config.LOAD_MODEL else [9999]
    train_losses = [9999]
    if config.LOAD_MODEL:
        print(valid_losses[0])
    wait = 0
    print('--'*20)
    start_time = time.time()
    for i, step in enumerate(range(config.STEP)):
        enc_scheduler.step()
        dec_scheduler.step()
        train_loss = run_step(config, enc, dec, train_loader, 
                              loss_function, enc_optimizer, dec_optimizer)
        if config.EMPTY_CUDA_MEMORY:
            torch.cuda.empty_cache()
            
        if config.NO_VALID:
            train_losses.append(train_loss)
            print('[{}/{}] (train) loss {:.4f} \n'.format(step+1, config.STEP, train_loss))
        else:
            valid_loss = validation(config, enc, dec, valid_loader, loss_function)
            if config.EMPTY_CUDA_MEMORY:
                torch.cuda.empty_cache()
            valid_losses.append(valid_loss)
            print('[{}/{}] (train) loss {:.4f} | (valid) loss {:.4f} \n'.format(
                    step+1, config.STEP, train_loss, valid_loss))

        # Save model
        if config.SAVE_MODEL:
            if config.SAVE_BEST:
                if config.NO_VALID:
                    if train_loss <= min(train_losses):
                        torch.save(enc.state_dict(), config.SAVE_ENC_PATH)
                        torch.save(dec.state_dict(), config.SAVE_DEC_PATH)
                        print('****** model saved updated! ******')
                else:
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
                
            # early stopping
            if config.NO_VALID:
                if train_loss > min(train_losses):
                    wait += 1
                    if wait > config.THRES:
                        print('****** early break!! ******')
                        break
            else:    
                if valid_loss > min(valid_losses):
                    wait += 1
                    if wait > config.THRES:
                        print('****** early break!! ******')
                        break

    end_time = time.time()
    total_time = end_time-start_time
    hour = int(total_time // (60*60))
    minute = int((total_time - hour*60*60) // 60)
    second = total_time - hour*60*60 - minute*60
    print('\nTraining Excution time with validation: {:d} h {:d} m {:.4f} s'.format(hour, minute, second))