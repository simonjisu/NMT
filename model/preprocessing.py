# coding utf-8
import sys
import random

if '-h' in sys.argv:
    temp = {'-h': 'help', 
            '-f': 'set filepath, default is "./data/translate"',
            '-sf': 'location of source file, default is "./data/en_de/tokenized.de"',
            '-tf': 'location of target file, default is "./data/en_de/tokenized.en"',
            '-tr': 'train sample rate, use it with "-va" arg. together',
            '-va': 'validation sample rate, use it with "-tr" arg. together',
            '-nosplit': 'do not split data',
            '-r' : 'random shuffle data',
            '-n' : 'choose n sentences',}
    
    print('=' * 30)
    for k, v in temp.items():
        print('{} : {}'.format(k, v))
    print('=' * 30)
    print('one of "-nosplit" or "-tr, -va" is necessary.')
    sys.exit()

if '-f' in sys.argv:
    idx = sys.argv.index('-f') + 1
    file_path = sys.argv[idx]
else:
    file_path = './data/translate'
    
if '-tr' in sys.argv:
    idx = sys.argv.index('-tr') + 1
    tr = sys.argv[idx]
else:
    tr = 0.8

if '-va' in sys.argv:
    idx = sys.argv.index('-va') + 1
    va = sys.argv[idx]
else:
    va = 0.1

te = 1. - tr - va
if tr + va + te != 1.:
    print("the sum of training & valid & test rate are not 1, check your setting.")
    sys.exit()

if '-sf' in sys.argv:
    idx = sys.argv.index('-sf') + 1
    so_path = sys.argv[idx]
else:
    so_path = './data/en_de/tokenized.de'

if '-tf' in sys.argv:
    idx = sys.argv.index('-tf') + 1
    so_path = sys.argv[idx]
else:
    ta_path = './data/en_de/tokenized.en'
    
    
def read_files(path):
    with open(path, 'r', encoding='utf-8') as file:
        data = file.read().splitlines()
    return data
          
def write_file(path, data):
    with open(path, 'w', encoding='utf-8') as file:
        for s, t in data:
            print(s + '\t' + t, file=file)
    return None

######### start preprocessing ########
so_data = read_files(so_path)
ta_data = read_files(ta_path)
total_data = list(zip(so_data, ta_data))

if '-r' in sys.argv:
    random.shuffle(total_data)

if '-n' in sys.argv:
    idx = sys.argv.index('-n') + 1
    n_sentences = int(sys.argv[idx])
    total_data = total_data[:n_sentences]

if ('-tr' in sys.argv) and ('-va' in sys.argv):
    total_len = len(total_data)
    train_idx = int(total_len*tr)
    valid_idx = train_idx + int((total_len - train_idx)*va)
    train_data = total_data[:train_idx]
    valid_data = total_data[train_idx:valid_idx]
    test_data = total_data[valid_idx:]

    for txt, d in zip(['_train', '_valid', '_test'], [train_data, valid_data, test_data]):
        write_file(file_path+txt+'.txt', d)
elif '-nosplit' in sys.argv:
    write_file(file_path+'.txt', total_data)
else:
    print('insert args: "-nosplit" or "-tr & -va" both')
    sys.exit()
print("done!")
