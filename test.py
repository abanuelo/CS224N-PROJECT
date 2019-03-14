# -*- coding: utf-8 -*-
import torch
from utils import get_data, batch_iter, sents2tensor
from BiLSTM_CRF import BiLSTM_CRF
from GRU_CRF import GRU_CRF
from old_BiLSTM_CRF import BiLSTM_CRF as old_BiLSTM_CRF

batch_size = 3
embed_size = 5
hidden_size = 4

torch.manual_seed(1)
data = get_data("./data/train_small.txt", "./data/train_small_output_gold.txt")

extra_chars = "παβσε"
thai_chars = "กขฃคฅฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรฤลฦวศษสหฬอฮฯะั าำิ ี ึ ื ุ ู ฺ฿เแโใไๅๆ็ ่ ้ ๊ ๋ ์ ํ ๎๐๑๒๓๔๕๖๗๘๙".replace(" ", "")
eng_chars = " !\"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}‘’~"
all_chars = extra_chars+thai_chars+eng_chars
char2id = {c:i for i,c in enumerate(list(all_chars))}
device = torch.device('cpu')
tag2id = {"0": 0, "1": 1, 'σ': 2, 'π': 3, 'ε':4}

tag2id_old = {"0": 0, "1": 1, 'σ': 2, 'ε': 3}


bilstm = BiLSTM_CRF(len(char2id), len(tag2id), embed_size, hidden_size, tag2id['σ'], tag2id['ε'], tag2id['π'])
gru = GRU_CRF(len(char2id), len(tag2id), embed_size, hidden_size, tag2id['σ'], tag2id['ε'], tag2id['π'])


for e in range(100):
    for sents, gold in batch_iter(data, batch_size):
        sents_tensor=sents2tensor(sents, char2id, char2id['π'], device)
        gold_tensor=sents2tensor(gold, tag2id, tag2id['π'], device)
        mask = 1-sents_tensor.data.eq(char2id['π']).float()
        score = bilstm(sents_tensor, gold_tensor, mask)
        loss = torch.mean(score)
        loss.backward()

        score_gru = gru(sents_tensor, gold_tensor, mask)
        print(score_gru)
        loss_gru = torch.mean(score_gru)
        loss_gru.backward()


        print('score BiLSTM: ',score, 'loss:', loss)
        print('######')
        print('score GRU: ',score_gru, 'loss:', loss_gru)

        print('---------------------------------------------------')





# sents = torch.zeros()
# mask = 1-sents.data.eq(char2id['π']).float()
# crf(sents, mask)
# print(score)

