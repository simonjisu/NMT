# coding utf-8
# import packages
import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data import Field, BucketIterator, TabularDataset
from NMTutils import evaluation

from decoder import Decoder
from encoder import Encoder

import numpy as np


def train(config):
    if config.NOCUDA:
        USE_CUDA = False
        DEVICE = -1
    else:
        USE_CUDA = torch.cuda.is_available()
        DEVICE = torch.cuda.current_device()

    print('cuda states: {}'.format(USE_CUDA))
    if USE_CUDA:
        torch.cuda.manual_seed(1234)
    else:
        torch.manual_seed(1234)

    SOURCE = Field(tokenize=str.split, use_vocab=True, init_token="<s>", eos_token="</s>", lower=True, 
                   include_lengths=True, batch_first=True)
    TARGET = Field(tokenize=str.split, use_vocab=True, init_token="<s>", eos_token="</s>", lower=True, 
                   batch_first=True)

    train_data, valid_data, test_data = \
        TabularDataset.splits(path=config.PATH, format='tsv', train=config.TRAIN_FILE,
                              validation=config.VALID_FILE, test=config.TEST_FILE,
                              fields=[('so', SOURCE), ('ta', TARGET)])

    SOURCE.build_vocab(train_data)
    TARGET.build_vocab(train_data)

    train_loader = BucketIterator(train_data, batch_size=config.BATCH, device=DEVICE,
                                  sort_key=lambda x: len(x.so), sort_within_batch=True, repeat=False)
    valid_loader = BucketIterator(valid_data, batch_size=config.BATCH, device=DEVICE,
                                  sort_key=lambda x: len(x.so), sort_within_batch=True, repeat=False)

    # parameters
    V_so = len(SOURCE.vocab)
    V_ta = len(TARGET.vocab)
    EARLY_STOPPING = False

    # build networks
    enc = Encoder(V_so, config.EMBED, config.HIDDEN, config.NUM_HIDDEN, bidrec=True, dropout_rate=config.DROPOUT_RATE,
                  layernorm=config.LAYERNORM, USE_CUDA=USE_CUDA)
    dec = Decoder(V_ta, config.EMBED, 2*config.HIDDEN, hidden_size2=config.HIDDEN2,
                  sos_idx=SOURCE.vocab.stoi['<s>'], method=config.METHOD, dropout_rate=config.DROPOUT_RATE,
                  decode_method=config.DECODE_METHOD, layernorm=config.LAYERNORM, USE_CUDA=USE_CUDA)
    if USE_CUDA:
        enc = enc.cuda()
        dec = dec.cuda()

    loss_function = nn.CrossEntropyLoss(ignore_index=TARGET.vocab.stoi['<pad>'])
    enc_optimizer = optim.Adam(enc.parameters(), lr=config.LR, weight_decay=config.LAMBDA)
    dec_optimizer = optim.Adam(dec.parameters(), lr=config.LR * config.DECLR, weight_decay=config.LAMBDA)

    if config.LR_SCH:
        enc_scheduler = optim.lr_scheduler.MultiStepLR(gamma=0.1, milestones=[30, 40, 50], optimizer=enc_optimizer)
        dec_scheduler = optim.lr_scheduler.MultiStepLR(gamma=0.1, milestones=[30, 40, 50], optimizer=dec_optimizer)
    else:
        enc_scheduler = optim.lr_scheduler.MultiStepLR(gamma=0.1,
                                                       milestones=[int(config.STEP / 4), int(config.STEP / 2),
                                                                   int(3 * config.STEP / 4)],
                                                       optimizer=enc_optimizer)
        dec_scheduler = optim.lr_scheduler.MultiStepLR(gamma=0.1,
                                                       milestones=[int(config.STEP / 4), int(config.STEP / 2),
                                                               int(3 * config.STEP / 4)],
                                                       optimizer=dec_optimizer)



    # train
    wait = 0
    valid_loss_list = [1e10] if config.EARLY else None
    enc.train()
    dec.train()
    for step in range(config.STEP):
        losses=[]
        enc_scheduler.step()
        dec_scheduler.step()
        if config.EARLY and EARLY_STOPPING:
            break
        for i, batch in enumerate(train_loader):
            inputs, lengths = batch.so
            targets = batch.ta

            enc.zero_grad()
            dec.zero_grad()

            output, hidden = enc(inputs, lengths.tolist())
            preds, _ = dec(hidden, output, lengths.tolist(), targets.size(1))  # max_len

            loss = loss_function(preds, targets.view(-1))
            losses.append(loss.item())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(enc.parameters(), 50.0)  # gradient clipping
            torch.nn.utils.clip_grad_norm_(dec.parameters(), 50.0)  # gradient clipping
            enc_optimizer.step()
            dec_optimizer.step()

        if step % config.EVAL_EVERY == 0:
            valid_losses = evaluation(enc, dec, loss_function, valid_loader)
            
            msg = '[{}/{}] train_loss: {:.4f}, valid_loss: {:.4f}'.format(\
              step+1, config.STEP, np.mean(losses), np.mean(valid_losses))
            print(msg)
            
            if config.EARLY:
                valid_loss_list.append(np.mean(valid_losses))
                diff = valid_loss_list[-2] - valid_loss_list[-1]
                if diff < -config.MIN_DELTA:
                    if wait > config.EARLY_PATIENCE:
                        EARLY_STOPPING = True
                        print("Early Stopping!")
                        print(valid_loss_list[1:])
                        break
                    else:
                        wait += 1


            losses = []
            enc.train()
            dec.train()

    # save model
    torch.save(enc.state_dict(), config.SAVE_ENC_PATH)
    torch.save(dec.state_dict(), config.SAVE_DEC_PATH)
