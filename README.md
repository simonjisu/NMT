# Neural Machine Translation

Paper Implementation: [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) - Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio (v7 2016)

## Getting Started

### Prerequisites

```
pytorch 0.4.0
argparse 1.1
numpy 1.14.3
matplotlib 2.2.2
```

### Demo

Not yet

### Data Preprocessing

After download data from data **source2** below `reference`, run `preprocessing2.py`

> `-h` : 'help', 
>
> `-r` : 'reverse',
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

### NMT Training argument parser

For 'HELP' please insert argument behind `main.py -h`. For example, 

```
python3 -u main.py -trp eng-fra-filtered.train \ 
                   -vap eng-fra-filtered.valid \ 
                   -tep eng-fra-filtered.test \ 
                   -svpe ./data/model/eng_fra/eng-fra16.enc \ 
                   -svpd ./data/model/eng_fra/eng-fra16.dec \ 
                   -el -elpat 3 -bat 256 -ee 3 -stp 90 \ 
                   -hid 512 -emd 256 -nhl 4 \ 
                   -wdk 0.0001 -drop -dropr 0.1
```

### argument parser

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
> `-lrsch` (LR_SCH) : use fixed learning schedule, default=False
> 
> `-declr` (DECLR) : decoder learning rate, type=float, default=5.0
>
> `-wdk` (LAMBDA) : L2 regularization, weight_decay in optimizer, type=float, default=0.0
>
> `-drop` (DROPOUT) : using dropout or not, default=False
>
> `-ee` (EVAL_EVERY) : eval every step size, type=int, default=1
>
> `-el` (EARLY) : using earlystopping, default=False
>
> `-elpat` (EARLY_PATIENCE) : earlystopping patience number, type=int, default=5
>
> `-elmin` (MIN_DELTA) : earlystopping minimum delta, type=float, default=0.0

## Trainlog

Train logs are in `trainlog` directory. 


## References

* arichitecture picture: https://arxiv.org/pdf/1703.03906.pdf
* tutorial: https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/seq2seq-translation-batched.ipynb
* data source: http://www.statmt.org/wmt14/translation-task.html
* data source2: http://www.manythings.org/anki/

## Todo:
* Layer Normalizaiton: https://discuss.pytorch.org/t/speed-up-for-layer-norm-lstm/5861
* seq2seq beam search: https://guillaumegenthial.github.io/sequence-to-sequence.html
* large output vocab problem: http://www.aclweb.org/anthology/P15-1001
* Recurrent Memory Networks(using Memory Block): https://arxiv.org/pdf/1601.01272
* BPE: https://arxiv.org/abs/1508.07909 

## License

This project is licensed under the MIT License 


