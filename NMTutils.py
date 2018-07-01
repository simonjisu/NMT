import argparse
from torchtext.data import Field, BucketIterator, TabularDataset
from collections import defaultdict
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# About model

def get_parser(lang1, lang2, data_path='./data/en_fa/', file_type='small', **kwargs):

    config_dict = dict(batch_size={'0': 1, '1': 64, '2': 128, '3': 256},
                       embed_size={'1': 300, '2': 256},
                       hidden_size={'1': 600, '2': 512},
                       hidden_layer={'3': 3, '4': 4, '5': 5, '6': 6},
                       drop_rate={'0': 0.0, '1': 0.5, '2': 0.1},
                       method={'1': 'general', '2': 'paper'})

    parser = argparse.ArgumentParser(description='NMT argument parser')
    parser.add_argument('-pth', '--PATH', help='location of path', type=str, default=data_path)
    parser.add_argument('-mth', '--METHOD', help='attention methods: dot, general, concat, paper', type=str, default=config_dict['method'][kwargs['method']])
    parser.add_argument('-bat', '--BATCH', help='batch size', type=int, default=config_dict['batch_size'][kwargs['batch_size']])
    parser.add_argument('-hid', '--HIDDEN', help='hidden size', type=int, default=config_dict['hidden_size'][kwargs['hidden_size']])
    parser.add_argument('-emd', '--EMBED', help='embed size', type=int, default=config_dict['embed_size'][kwargs['embed_size']])
    parser.add_argument('-nhl', '--NUM_HIDDEN', help='number of hidden layers', type=int, default=config_dict['hidden_layer'][kwargs['hidden_layer']])
    parser.add_argument('-dropr', '--DROPOUT_RATE', help='using dropout rate, 0 means not use drop out', type=float, default=config_dict['drop_rate'][kwargs['drop_rate']])
    parser.add_argument('-ee', '--EVAL_EVERY', help='eval every step size', type=int, default=1)
    parser.add_argument('-hid2', '--HIDDEN2', help='hidden size used for attention(not nessesary)', type=int, default=None)

    if lang2 == 'kor':
        parser.add_argument('-trp', '--TRAIN_FILE', help='location of training path', type=str, default='{}-{}.train'.format(lang1, lang2))
        parser.add_argument('-vap', '--VALID_FILE', help='location of valid path', type=str, default='{}-{}.valid'.format(lang1, lang2))
        parser.add_argument('-tep', '--TEST_FILE', help='location of test path', type=str, default='{}-{}.test'.format(lang1, lang2))
    elif lang2 == 'fra':
        parser.add_argument('-trp', '--TRAIN_FILE', help='location of training path', type=str, default='{}-{}-{}.train'.format(lang1, lang2, file_type))
        parser.add_argument('-vap', '--VALID_FILE', help='location of valid path', type=str, default='{}-{}-{}.valid'.format(lang1, lang2, file_type))
        parser.add_argument('-tep', '--TEST_FILE', help='location of test path', type=str, default='{}-{}-{}.test'.format(lang1, lang2, file_type))

    config = parser.parse_known_args()[0]
    return config


def build_data(config, device=-1):
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

    test_loader = BucketIterator(test_data, batch_size=config.BATCH, device=device,
                                 sort_key=lambda x: len(x.so), sort_within_batch=True, repeat=False)


    return SOURCE, TARGET, train_data, valid_data, test_data, test_loader


def get_model_config(modelcode, lang1, lang2, file_type='small', file_path='./data/en_fa/', device=-1):
    """
    insert modelcode in fixed string at right order below
    ** modelcode option **
    batch_size={'0': 1, '1': 64, '2': 128, '3': 256},
    embed_size={'1': 300, '2': 256},
    hidden_size={'1': 600, '2': 512},
    hidden_layer={'3': 3, '4': 4, '5': 5, '6': 6},
    drop_rate={'0': 0.0, '1': 0.5, '2': 0.1},
    method={'1': 'general', '2': 'paper'},
    """
    config = get_parser(lang1, lang2, file_path=file_path, file_type=file_type,
                        batch_size=modelcode[0],
                        embed_size=modelcode[1],
                        hidden_size=modelcode[2],
                        hidden_layer=modelcode[3],
                        drop_rate=modelcode[4],
                        method=modelcode[5])
    SOURCE, TARGET, train_data, valid_data, test_data, test_loader = build_data(config, device)
    return config, test_data, test_loader, SOURCE, TARGET


# About metrics

def evaluation(enc, dec, loss_function, loader):
    enc.eval()
    dec.eval()
    valid_losses = []

    for i, batch in enumerate(loader):
        inputs, lengths = batch.so
        targets = batch.ta

        output, hidden = enc(inputs, lengths.tolist())
        preds, _ = dec(hidden, output, lengths.tolist(), targets.size(1))  # max_len

        loss = loss_function(preds, targets.view(-1))
        valid_losses.append(loss.item())

    return valid_losses


class BLEU(object):
    def __init__(self, test_data, filter_num=1, smooth=None):
        """
        args:
        * test data: Field class in torchtext
        * filter_num: filter counts of target sentences
        * smooth: smooth method, see https://www.nltk.org/_modules/nltk/translate/bleu_score.html
        """
        self.test_data = test_data
        self.so_ret_dict = defaultdict(list)
        self.get_source_reference(test_data, filter_num=filter_num)
        if smooth is not None:
            assert int(smooth) in list(range(6)), "only can use 6 method in nltk SmoothingFunction class"
            self.smooth = smooth
            self.smooth_f = SmoothingFunction()

    def get_source_reference(self, data, filter_num=1):
        for d in data.examples:
            src = ' '.join(d.so)
            tar = ' '.join(d.ta)
            self.so_ret_dict[src].append(tar)

        self.so_ret_dict = {k: v for k, v in self.so_ret_dict.items() if len(v) >= filter_num}

    def evaluation_bleu(self, soruce_sentence, hypotheses, get_ref=False):
        """
        hypotheses: a tokenized list
        """
        assert isinstance(hypotheses, list), 'hypothesis must be a tokenized list'

        references = self.so_ret_dict[soruce_sentence]
        score = sentence_bleu([x.split() for x in references], hypotheses,
                              smoothing_function=self.smooth_f.__getattribute__('method{}'.format(self.smooth)))
        if get_ref:
            return score, references
        return score



def show_attention(input_words, output_words, attentions):
    """
    borrowed code from https://github.com/spro/practical-pytorch/blob/master/seq2seq-translation/seq2seq-translation-batched.ipynb
    """
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(attentions.numpy(), cmap='bone')
    fig.colorbar(cax)

    # Set up axes
    ax.set_xticklabels([''] + input_words, rotation=90)
    ax.set_yticklabels([''] + output_words)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

#     show_plot_visdom()
    plt.show()
    plt.close()