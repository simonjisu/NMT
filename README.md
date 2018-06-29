# Neural Machine Translation

Paper Implementation: [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) - Dzmitry Bahdanau, Kyunghyun Cho, Yoshua Bengio (v7 2016)

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on how to deploy the project on a live system.

### Prerequisites

```
pytorch 0.4.0
argparse 1.1
numpy 1.14.3
matplotlib 2.2.2
```

### Demo

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

Train logs are in `trainlog` directory. Trained & valid & test for 50000 data sets

Following table is hyperparameteres i've tried to train, `loss` is validation loss, `(el)` beside the `loss` is the early stopped step.

See [Notebook]() for test sets.

|trainID|loss(el)|bat|dropr|emd|hid|nhl|wdk|stp|ee|el|mth|lrsch|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|1|3.6691|64|0.0|300|600|3|0|30|1|F|"general"|STEP\*(1/2)|
|2|3.1080|64|0.5|300|600|3|0|30|1|F|"general"|STEP\*(1/2)|
|3|2.3471|64|0.0|300|600|3|0.0001|30|1|F|"general"|STEP\*(1/2)|
|4|2.2082|64|0.5|300|600|3|0.0001|30|1|F|"general"|STEP\*(1/2)|
|5|2.2157|64|0.5|300|600|3|0.0001|30|1|F|"general"|STEP\*(1/2, 3/4)|
|6|2.1632|64|0.5|300|600|3|0.0001|50|1|F|"general"|STEP\*(1/2, 3/4)|
|7|2.0926(24)|64|0.5|300|600|3|0.0001|40|1|T|"general"|STEP\*(1/4, 1/2, 3/4)|
|8|2.1934(22)|64|0.1|256|512|4|0.0001|40|1|T|"general"|STEP\*(1/4, 1/2, 3/4)|
|9|2.3073(22)|64|0.1|256|512|5|0.0001|40|1|T|"general"|STEP\*(1/4, 1/2, 3/4)|
|10|3.0452|64|0.1|256|512|6|0.0001|40|1|F|"general"|STEP\*(1/4, 1/2, 3/4)|
|11|2.1887(21)|64|0.1|256|512|6|0.0001|40|1|T|"general"|STEP\*(1/4, 1/2, 3/4)|
|12|2.2009(40))|128|0.1|256|512|4|0.0001|40|2|T|"general"|STEP\*(1/4, 1/2, 3/4)|
|13|2.3547(34)|128|0.1|256|512|5|0.0001|60|3|T|"paper"|STEP\*(1/4, 1/2, 3/4)|
|14|2.3505(56)|128|0.1|256|512|4|0.0001|100|5|T|"general"|STEP\*(1/4, 1/2, 3/4)|

## Deployment

### References

* arichitecture picture: https://arxiv.org/pdf/1703.03906.pdf
* tutorial: https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/seq2seq-translation-batched.ipynb
* data source: http://www.statmt.org/wmt14/translation-task.html
* data source2: http://www.manythings.org/anki/

### Todo:

* BLEU score
* Layer Normalizaiton: https://discuss.pytorch.org/t/speed-up-for-layer-norm-lstm/5861
* seq2seq beam search: https://guillaumegenthial.github.io/sequence-to-sequence.html

## License

This project is licensed under the MIT License 


