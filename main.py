import argparse
from train import train

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='NMT argument parser')
    parser.add_argument('-pth', '--PATH', help='location of path', type=str, default='./data/en_fa/')
    parser.add_argument('-trp', '--TRAIN_FILE', help='location of training path', type=str, default='eng-fra-small.train')
    parser.add_argument('-vap', '--VALID_FILE', help='location of valid path', type=str, default='eng-fra-small.valid')
    parser.add_argument('-tep', '--TEST_FILE', help='location of test path', type=str, default='eng-fra-small.test')
    parser.add_argument('-mth', '--METHOD', help='attention methods: dot, general, concat, paper', type=str, default='general')
    parser.add_argument('-svpe', '--SAVE_ENC_PATH', help='saving encoder model path', type=str, default='./data/model/fra_eng.enc')
    parser.add_argument('-svpd', '--SAVE_DEC_PATH', help='saving decoder model path', type=str, default='./data/model/fra_eng.dec')
    parser.add_argument('-bat', '--BATCH', help='batch size', type=int, default=128)
    parser.add_argument('-hid', '--HIDDEN', help='hidden size', type=int, default=600)
    parser.add_argument('-hid2', '--HIDDEN2', help='hidden size used for attention(not nessesary)', type=int, default=None)
    parser.add_argument('-emd', '--EMBED', help='embed size', type=int, default=300)
    parser.add_argument('-stp', '--STEP', help='number of iteration', type=int, default=200)
    parser.add_argument('-nhl', '--NUM_HIDDEN', help='number of hidden layers', type=int, default=3)
    parser.add_argument('-lr', '--LR', help='learning rate', type=float, default=0.001)

    config = parser.parse_args()
    print(config)
    train(config)