import math
import time
import sys
from docopt import docopt
import torch
from itertools import zip_longest
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import numpy as np
from f1 import compute_F1_scores
#import reader
from BiLSTM_CRF import BiLSTM_CRF
from GRU_CRF import GRU_CRF
from utils import batch_iter, get_data, sents2tensor


# extra_chars = "παβσε"
# thai_chars = "กขฃคฅฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรฤลฦวศษสหฬอฮฯะั าำิ ี ึ ื ุ ู ฺ฿เแโใไๅๆ็ ่ ้ ๊ ๋ ์ ํ ๎๐๑๒๓๔๕๖๗๘๙".replace(" ", "")
# eng_chars = " !\"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}‘’~"
# all_chars = extra_chars+thai_chars+eng_chars
# char2id = {c:i for i,c in enumerate(list(all_chars))}
# device = torch.device('cpu')
# tag2id = {"0": 0, "1": 1, 'σ': 2, 'π': 3, 'ε':4}


# f = open('./data/problems.txt')
# inp = f.readlines()[0].strip('\n')

# model = BiLSTM_CRF.load('./models/model.bin')
# inp_tensor=sents2tensor(inp, char2id, char2id['π'], device)
# mask = 1-inp_tensor.data.eq(char2id['π']).float()
# model_output = model.decode(inp_tensor, mask)
# print(model_output)


print("GRU medium")
F1_micro, F1_macro = compute_F1_scores('./outputs/test_outputs_gru_medium.txt', './data/test_output_gold.txt')
print("F1_micro", F1_micro)
print("F1_macro", F1_macro)
print('-'*40)

print("LSTM medium")
F1_micro, F1_macro = compute_F1_scores('./outputs/test_outputs_medium.txt', './data/test_output_gold.txt')
print("F1_micro", F1_micro)
print("F1_macro", F1_macro)
print('-'*40)

print("NE LSTM medium")
F1_micro, F1_macro = compute_F1_scores('./outputs/test_outputs_ne_medium.txt', './data/test_output_gold_ne.txt')
print("F1_micro", F1_micro)
print("F1_macro", F1_macro)
print('-'*40)

