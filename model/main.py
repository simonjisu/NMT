import os
import argparse
import torch
from train import import_data, build_model, train_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NMT argument parser')
    # data
    parser.add_argument('-root', '--ROOTPATH', help='location of path', type=str, default='../data/')
    parser.add_argument('-dt', '--DATATYPE', help='datatype for training', type=str, default='iwslt') 
    parser.add_argument('-stp', '--STEP', help='Trainging Steps', type=int, default=10)
    parser.add_argument('-bs', '--BATCH', help='Batch Size', type=int, default=64)
    parser.add_argument('-cuda', '--USE_CUDA', help='Use cuda if exists', action='store_true')
    parser.add_argument('-emptymem', '--EMPTY_CUDA_MEMORY', help='Use cuda empty cashce', action='store_true')
    parser.add_argument('-minfreq', '--MIN_FREQ', help='Minmum frequence of vocab', type=int, default=2)
    parser.add_argument('-maxlen', '--MAX_LEN', help='Max length of sentences in dataset', type=int, default=100)
    
    # model
    parser.add_argument('-hid', '--HIDDEN', help='hidden size', type=int, default=600)
    parser.add_argument('-hid2', '--HIDDEN2', help='hidden size used for attention(not nessesary)', type=int, default=None)
    parser.add_argument('-emd', '--EMBED', help='embed size', type=int, default=300)
    parser.add_argument('-nhl', '--NUM_HIDDEN', help='number of hidden layers', type=int, default=3)
    parser.add_argument('-mth', '--METHOD', help='attention methods: dot, general, concat, paper', type=str, default='general')
    parser.add_argument('-drop', '--DROPOUT_RATE', help='using dropout rate, 0 means not use drop out', type=float, default=0.0)
    parser.add_argument('-lnorm', '--LAYERNORM', help='use layer normalization', action='store_true')
    parser.add_argument('-rtw', '--RETURN_W', help='Return weight', action='store_true')
    # optimizer
    parser.add_argument('-lr', '--LR', help='learning rate', type=float, default=0.001)
    parser.add_argument('-declr', '--DECLR', help='decoder learning rate', type=float, default=5.0)
    parser.add_argument('-wdk', '--LAMBDA', help='L2 regularization, weight_decay in optimizer', type=float, default=0.0)
    
    # save model
    parser.add_argument('-save', '--SAVE_MODEL', help='Save model', action='store_true')
    parser.add_argument('-savebest', '--SAVE_BEST', help='Save best model', action='store_true')
    parser.add_argument('-svpe', '--SAVE_ENC_PATH', help='saving encoder model path', type=str, default='./saved_models/iswlt.enc')
    parser.add_argument('-svpd', '--SAVE_DEC_PATH', help='saving decoder model path', type=str, default='./saved_models/iswlt.dec')
    
    # others 
    parser.add_argument('-pee', '--PRINT_EVERY', help='print every step size', type=int, default=1)
    parser.add_argument('-thres', '--THRES', help='earlystopping patience number', type=int, default=5)

    config = parser.parse_args()
    print(config)
    if config.USE_CUDA:
        assert config.USE_CUDA == torch.cuda.is_available(), 'cuda is not avaliable.'
    DEVICE = 'cuda' if config.USE_CUDA else None
    SRC, TRG, train, valid, train_loader, valid_loader = import_data(config, device=DEVICE)
    enc, dec, loss_function, enc_optimizer, dec_optimizer, enc_scheduler, dec_scheduler = \
        build_model(config, src_field=SRC, trg_field=TRG, device=DEVICE)
    
    train_model(config, enc, dec, loss_function, enc_optimizer, dec_optimizer, enc_scheduler, dec_scheduler, train_loader, valid_loader)