import argparse
import torch
import torch.nn as nn
import random

from decoder import Decoder
from encoder import Encoder

from torchtext.data import Field, BucketIterator, TabularDataset
from collections import defaultdict
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


