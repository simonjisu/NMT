import os
import argparse
import torch
from train import import_data, build_model, train_model

def build_config_file(config_path='./settings.py'):
    with open('./runtrain.sh', 'r') as file:
        data = file.read().splitlines()
    log_file = data[-1].split('>')[1].strip()[1:-2]
    with open(log_file, 'r', encoding='utf-8') as file:
        data = file.read().splitlines()[0][10:-1]
        data = [x.strip() for x in data.split(',')]
        for i, x in enumerate(data):
            if ('SAVE' in x) & ('_PATH' in x):
                data[i] = x.split("'.")[0] +"'./model" + x.split("'.")[1]
            elif 'ROOTPATH' in x:
                data[i] = "ROOTPATH='./data/'"
            elif 'RETURN_W' in x:
                data[i] = "RETURN_W=True"
        with open(config_path, 'w', encoding='utf-8') as f:
            for x in data:
                print(x, file=f)


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
    parser.add_argument('-maxlen', '--MAX_LEN', help='Max length of sentences in dataset', type=int, default=50)
    
    
    # model
    parser.add_argument('-hid', '--HIDDEN', help='hidden size', type=int, default=600)
    parser.add_argument('-emd', '--EMBED', help='embed size', type=int, default=300)
    parser.add_argument('-enhl', '--ENC_N_LAYER', help='number of hidden layers', type=int, default=3)
    parser.add_argument('-dnhl', '--DEC_N_LAYER', help='number of hidden layers', type=int, default=3)
    parser.add_argument('-mth', '--METHOD', help='attention methods: dot, general, concat, paper', type=str, default='general')
    parser.add_argument('-drop', '--DROP_RATE', help='using dropout rate, 0 means not use drop out', type=float, default=0.0)
    parser.add_argument('-lnorm', '--L_NORM', help='use layer normalization, cannot use with dropout at same time', action='store_true')
    parser.add_argument('-rtw', '--RETURN_W', help='Return weight', action='store_true')
    parser.add_argument('-tf', '--TF', help='Teacher Forcing Learning', action='store_true')
    # optimizer
    parser.add_argument('-optim', '--OPTIM', help='optimizer methods', type=str, default='adam')
    parser.add_argument('-lr', '--LR', help='learning rate', type=float, default=0.001)
    parser.add_argument('-declr', '--DECLR', help='decoder learning rate', type=float, default=5.0)
    parser.add_argument('-wdk', '--LAMBDA', help='L2 regularization, weight_decay in optimizer', type=float, default=0.0)
    
    
    # save model
    parser.add_argument('-novalid', '--NO_VALID', help='no validation', action='store_true')
    parser.add_argument('-save', '--SAVE_MODEL', help='Save model', action='store_true')
    parser.add_argument('-savebest', '--SAVE_BEST', help='Save best model', action='store_true')
    parser.add_argument('-svpe', '--SAVE_ENC_PATH', help='saving encoder model path', type=str, default='./saved_models/iswlt.enc')
    parser.add_argument('-svpd', '--SAVE_DEC_PATH', help='saving decoder model path', type=str, default='./saved_models/iswlt.dec')
    # load model: training from loaded model
    parser.add_argument('-load', '--LOAD_MODEL', help='Save model', action='store_true')
    parser.add_argument('-ldpe', '--LOAD_ENC_PATH', help='saving encoder model path', type=str, default='./saved_models/iswlt.enc')
    parser.add_argument('-ldpd', '--LOAD_DEC_PATH', help='saving encoder model path', type=str, default='./saved_models/iswlt.dec')
    # others 
    parser.add_argument('-pee', '--PRINT_EVERY', help='fraction of print every in len(trainloader)', type=int, default=1)
    parser.add_argument('-thres', '--THRES', help='earlystopping patience number', type=int, default=5)

    config = parser.parse_args()
    print(config)
    if config.USE_CUDA:
        assert config.USE_CUDA == torch.cuda.is_available(), 'cuda is not avaliable.'
    DEVICE = 'cuda' if config.USE_CUDA else None
    SRC, TRG, train, valid, train_loader, valid_loader = import_data(config, device=DEVICE, is_test=False)
    enc, dec, loss_function, enc_optimizer, dec_optimizer, enc_scheduler, dec_scheduler = \
        build_model(config, src_field=SRC, trg_field=TRG, device=DEVICE)
    if config.LOAD_MODEL:
        enc.load_state_dict(torch.load(config.LOAD_ENC_PATH))
        dec.load_state_dict(torch.load(config.LOAD_DEC_PATH))
        print("Load complete!")
    
    train_model(config, enc, dec, loss_function, enc_optimizer, dec_optimizer, enc_scheduler, dec_scheduler, train_loader, valid_loader)
    build_config_file()