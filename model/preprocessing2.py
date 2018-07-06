# coding utf-8
import sys
import re
import random
import unicodedata
from konlpy.tag import Twitter

if '-h' in sys.argv:
    temp = {'-h': 'help', 
            '-r': 'reverse',
            '-rand': 'random shuffle',
            '-wf': 'set filepath, default is "./data/translate"',
            '-rf': 'set filepath, default is "./data/en_fa/eng-fra.txt"',
            '-train': 'train sample rate, default is 0.8',
            '-test': 'test sample rate, default is 0.2',
            '-valid': 'validation sample rate, default is None',
            '-min_w': 'filter minimum words of a sentence, default is 3, have to insert if you want to filter',
            '-max_w': 'filter maximum words of a sentence, default is 25, have to insert if you want to filter',
            '-n': 'filter n paris of sentences',
            '-tagger': 'tokenizer for languages default is "en", can try "kr'}
    
    print('=' * 30)
    for k, v in temp.items():
        print('{} : {}'.format(k, v))
    print('=' * 30)
    sys.exit()
    
if '-rf' in sys.argv:
    idx = sys.argv.index('-rf') + 1
    READ_PATH = sys.argv[idx]
else:
    READ_PATH = './data/en_fa/eng-fra.txt'

if '-wf' in sys.argv:
    idx = sys.argv.index('-wf') + 1
    WRITE_PATH = sys.argv[idx]
else:
    WRITE_PATH = './data/translate'

if '-train' in sys.argv:
    idx = sys.argv.index('-train') + 1
    TRAIN = float(sys.argv[idx])
else:
    TRAIN = 0.8

if '-test' in sys.argv:
    idx = sys.argv.index('-test') + 1
    TEST = float(sys.argv[idx])
else:
    TEST = 0.2

if '-valid' in sys.argv:
    idx = sys.argv.index('-valid') + 1
    VALID = float(sys.argv[idx])
else:
    VALID = None
    
if '-min_w' in sys.argv:
    idx = sys.argv.index('-min_w') + 1
    MIN_WORD = int(sys.argv[idx])
else:
    MIN_WORD = 3

if '-max_w' in sys.argv:
    idx = sys.argv.index('-max_w') + 1
    MAX_WORD = int(sys.argv[idx])
else:
    MAX_WORD = 25

if '-n' in sys.argv:
    idx = sys.argv.index('-n') + 1
    N_PAIRS = int(sys.argv[idx])
else:
    N_PAIRS = 50000

if '-tagger' in sys.argv:
    idx = sys.argv.index('-tagger') + 1
    TAGGER = sys.argv[idx]
else:
    TAGGER = 'en'

# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s):
    return ''.join( c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn' )


# Lowercase, trim, and remove non-letter characters
def normalize_string(s):
    s = unicode_to_ascii(s.lower().strip())
    # s = re.sub(r"([,.!?])", r" \1 ", s)
    # s = re.sub(r"[^a-zA-Z,.!?]+", r" ", s)
    s = re.sub(r"([,.!?\"\'\-])", r" \1 ", s)
    s = re.sub(r"[^a-zA-Z,.!?\"\'\-]+", r" ", s)
    s = re.sub(r"\s+", r" ", s).strip()
    return s


def read_files(path, tagger=None):
    with open(path, 'r', encoding='utf-8') as file:
        pairs = [d.split('\t') for d in file.read().splitlines()]
        if tagger:
            pairs = [[normalize_string(s).split(), tagger.morphs(t)] for s, t in pairs]
        else:
            pairs = [[normalize_string(s).split(), normalize_string(t).split()] for s, t in pairs]
    return pairs


def write_file(data, path):
    print('writing file: {}'.format(path.split('/')[-1]))
    with open(path, 'w', encoding='utf-8') as file:
        for s, t in data:
            print(' '.join(s) + '\t' + ' '.join(t), file=file)
    

def filter_pairs(pairs, min_words, max_words):
    filtered_pairs = []
    for s, t in pairs:
        if (len(s) >= min_words and len(s) <= max_words) and (len(t) >= min_words and len(t) <= max_words):
            filtered_pairs.append([s, t])
    return filtered_pairs


def train_test_split(data, train=0.8, valid=None, test=0.1):
    if valid is None:
        assert (train + test == 1.) and (train > 0) and (test > 0), 'train ratio + valid ratio + test ratio must be 1'
    else:
        assert (train + valid + test == 1.) and (train*valid*test > 0), 'train ratio + valid ratio + test ratio must be 1'
        
    train_idx = int(len(data)*train)
    test_idx = int(len(data)*valid)
    if valid:
        train_data = data[:train_idx]
        valid_data = data[train_idx:-test_idx]
        test_data = data[-test_idx:]
        return train_data, valid_data, test_data
    else:
        train_data = data[:train_idx]
        test_data = data[train_idx:]
        return train_data, test_data


def main():
    """
    run codes!
    """
    tagger = Twitter() if 'kr' in sys.argv else None

    pairs = read_files(READ_PATH, tagger=tagger)
    if '-min_w' in sys.argv and '-max_w' in sys.argv:
        pairs = filter_pairs(pairs, MIN_WORD, MAX_WORD)
    
    if '-r' in sys.argv:
        pairs = [[t, s] for s, t in pairs]

    if '-rand' in sys.argv:
        random.shuffle(pairs)

    if '-n' in sys.argv:
        pairs = pairs[:N_PAIRS]
    
    if ('-train' in sys.argv and '-test' in sys.argv) or ('-train' in sys.argv and '-valid' in sys.argv and '-test' in sys.argv):
        datas = train_test_split(pairs, train=TRAIN, valid=VALID, test=TEST)
    
    if len(datas) == 2:
        str_list = ['.train', '.test']
    else:
        str_list = ['.train', '.valid', '.test']
        
    for s, d in zip(str_list, datas):
        write_file(d, WRITE_PATH+s)
        
    print('done!')


if __name__ == "__main__":
    main()