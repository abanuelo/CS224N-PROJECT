# -*- coding: utf-8 -*-
"""
run.py
-------------------
This file will initialize the dataset, character lookup table
and run training/tsting
"""



import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import reader


#####################################################################
# Run training


torch.manual_seed(1)

START_TAG = "σ"
STOP_TAG = "ε"
EMBEDDING_DIM = 5
HIDDEN_DIM = 4

#Load training data
training_data = reader.return_training()


def getDictionary():
	#load the dictionary
	thai_chars = "กขฃคฅฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรฤลฦวศษสหฬอฮฯะั าำิ ี ึ ื ุ ู ฺ฿เแโใไๅๆ็ ่ ้ ๊ ๋ ์ ํ ๎๐๑๒๓๔๕๖๗๘๙".replace(" ", "")
	eng_chars = " !\"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}"
	extra_chars = "αβσε"
	all_chars = thai_chars+eng_chars+extra_chars
	char2ix = {c:i for i,c in enumerate(list(all_chars))}
	ix2char = {i:c for i,c in enumerate(list(all_chars))}
	return char2ix, ix2char

def getTags():
	tag2ix = {"0": 0, "1": 1, START_TAG: 2, STOP_TAG: 3}
	return tag2ix

model = BiLSTM_CRF(len(char2ix), tag2ix, EMBEDDING_DIM, HIDDEN_DIM)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

# Check predictions before training
with torch.no_grad():
    precheck_sent = prepare_sequence(training_data[0][0], char2ix)
    precheck_tags = torch.tensor([tag2ix[t] for t in training_data[0][1]], dtype=torch.long)
    print(model(precheck_sent))

# Make sure prepare_sequence from earlier in the LSTM section is loaded
for epoch in range(
        10):  # again, normally you would NOT do 300 epochs, it is toy data
    for sentence, tags in training_data:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is,
        # turn them into Tensors of word indices.
        sentence_in = prepare_sequence(sentence, char2ix)
        targets = torch.tensor([tag2ix[t] for t in tags], dtype=torch.long)

        # Step 3. Run our forward pass.
        loss = model.neg_log_likelihood(sentence_in, targets)

        # Step 4. Compute the loss, gradients, and update the parameters by
        # calling optimizer.step()
        print('epoch: ' +str(epoch)+' loss: ' +str(loss))
        loss.backward()
        optimizer.step()


# Check predictions after training
with torch.no_grad():
    precheck_sent = prepare_sequence(training_data[0][0], char2ix)
    print(model(precheck_sent))
# We got it!