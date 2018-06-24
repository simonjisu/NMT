# NMT Study

1. Paper Implementation: [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) - Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio (v7 2016)

**references**
* tutorial: https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/seq2seq-translation-batched.ipynb
* tutorial2: https://github.com/DSKSD/DeepNLP-models-Pytorch/blob/master/notebooks/07.Neural-Machine-Translation-with-Attention.ipynb
* data source: http://www.statmt.org/wmt14/translation-task.html
* data source2: http://www.manythings.org/anki/

## Demo

## Data Preprocessing

After download data from data **source2**, run `preprocessing2.py`

> `-h` : 'help', 
>
> `-r` : 'random shuffle data',
>
> `-wf` : 'set filepath, default is "./data/translate"',
>
> `-rf` : 'set filepath, default is "./data/en_fa/eng-fra.txt"',
>
> `-train` : 'train sample rate, default is 0.8',
>
> `-test` : 'test sample rate, default is 0.2',
>
> `-valid` : 'validation sample rate, default is None',
>
> `-min_w` : 'filter minimum words of a sentence, default is 3, have to insert if you want to filter',
>
> `-max_w` : 'filter maximum words of a sentence, default is 25, have to insert if you want to filter',
>
> `-n` : 'filter n paris of sentences'

## NMT argument parser

For 'HELP' please insert argument behind `main.py -h`

> `-pth` (PATH) : location of path, type=str, default='./data/en_fa/'
>
> `-trp` (TRAIN_FILE) : location of training path, type=str, default='eng-fra-small.train'
>
> `-vap` (VALID_FILE) : location of valid path, type=str, default='eng-fra-small.valid'
>
> `-tep` (TEST_FILE) : location of test path, type=str, default='eng-fra-small.test'
>
> `-mth` (METHOD) : attention methods:  dot, general, concat, paper, type=str, default='general'
>
> `-svpe` (SAVE_ENC_PATH) : saving encoder model path, type=str, default='./model/fra_eng.enc'
>
> `-svpd` (SAVE_DEC_PATH) : saving decoder model path, type=str, default='./model/fra_eng.dec'
>
> `-bat` (BATCH) : batch size, type=int, default=64
>
> `-hid` (HIDDEN) : hidden size, type=int, default=600
>
> `-hid2` (HIDDEN2) : hidden size used for attention(not nessesary, type=int, default=None
>
> `-emd` (EMBED) : embed size, type=int, default=300
>
> `-stp` (STEP) : number of iteration, type=int, default=30
>
> `-nhl` (NUM_HIDDEN) : number of hidden layers, type=int, default=3
>
> `-lr` (LR) : learning rate, type=float, default=0.001
>
> `-declr` (DECLR) : decoder learning rate, type=float, default=5.0
>
> `-wdk` (LAMBDA) : L2 regularization, weight_decay in optimizer, type=float, default=0.0
>
> `-drop` (DROPOUT) : using dropout or not, action='store_true, default=False
>
> `-ee` (EVAL_EVERY) : eval every step size, type=int, default=1
>
> `-el` (EARLY) : using earlystopping, action='store_true, default=False
>
> `-elpat` (EARLY_PATIENCE) : earlystopping patience number, type=int, default=5
>
> `-elmin` (MIN_DELTA) : earlystopping minimum delta, type=float, default=0.0

## 

