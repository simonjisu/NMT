import argparse

def get_parser(lang1, lang2, batch_size=64, embed_size=256, hidden_size=512, hidden_layer=3, drop_rate=0.5, method='general'):
    parser = argparse.ArgumentParser(description='NMT argument parser')
    parser.add_argument('-pth', '--PATH', help='location of path', type=str, default='./data/en_fa/')
    parser.add_argument('-trp', '--TRAIN_FILE', help='location of training path', type=str, default='{}-{}-small.train'.format(lang1, lang2))
    parser.add_argument('-vap', '--VALID_FILE', help='location of valid path', type=str, default='{}-{}-small.valid'.format(lang1, lang2))
    parser.add_argument('-tep', '--TEST_FILE', help='location of test path', type=str, default='{}-{}-small.test'.format(lang1, lang2))
    parser.add_argument('-mth', '--METHOD', help='attention methods: dot, general, concat, paper', type=str, default=method)
    parser.add_argument('-bat', '--BATCH', help='batch size', type=int, default=batch_size)
    parser.add_argument('-hid', '--HIDDEN', help='hidden size', type=int, default=hidden_size)
    parser.add_argument('-hid2', '--HIDDEN2', help='hidden size used for attention(not nessesary)', type=int, default=None)
    parser.add_argument('-emd', '--EMBED', help='embed size', type=int, default=embed_size)
    parser.add_argument('-nhl', '--NUM_HIDDEN', help='number of hidden layers', type=int, default=hidden_layer)
    parser.add_argument('-drop', '--DROPOUT', help='using dropout or not', action='store_true', default=False)
    parser.add_argument('-dropr', '--DROPOUT_RATE', help='using dropout rate', type=float, default=drop_rate)
    parser.add_argument('-ee', '--EVAL_EVERY', help='eval every step size', type=int, default=1)

    config = parser.parse_known_args()[0]
    return config