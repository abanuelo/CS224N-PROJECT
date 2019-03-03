#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Usage:
    run.py train
    run.py train --train-input=<file> --train-gold=<file> [options]
    run.py test --test-input=<file> --test-gold=<file> [options]

Options:
    -d --debug                    Enable Debug mode
    --train-input=<file>          Training input path
    --train-gold=<file>           Training gold path
    --test-input=<file>           Testing input path
    --test-gold=<file>            Testing gold path
    --batch-size=<b>
    --valid-niter=<i>
    --max-epoch=<e>

This file will initialize the dataset, character lookup table
and run training/tsting           

"""
import sys
from docopt import docopt
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim
import reader
from  BiLSTM_CRF import BiLSTM_CRF, prepare_sequence

#####################################################################
# Run training

def getDictionary():
	"""
    Creates a lookup table for all characters
    Greek characters are used as special symbols
        (abbreviations, named entities, start, and stop)
    """
	thai_chars = "กขฃคฅฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรฤลฦวศษสหฬอฮฯะั าำิ ี ึ ื ุ ู ฺ฿เแโใไๅๆ็ ่ ้ ๊ ๋ ์ ํ ๎๐๑๒๓๔๕๖๗๘๙".replace(" ", "")
	eng_chars = " !\"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}"
	extra_chars = "αβσε"
	all_chars = thai_chars+eng_chars+extra_chars
	char2ix = {c:i for i,c in enumerate(list(all_chars))}
	ix2char = {i:c for i,c in enumerate(list(all_chars))}
	return char2ix, ix2char


def checkPredictions(training_data, char2ix, tag2ix):
    # Check predictions before training
    with torch.no_grad():
        precheck_sent = prepare_sequence(training_data[0][0], char2ix)
        print(model(precheck_sent))

def train(args:dict):
    ##########PARAMETER SETTINGS########
    START_TAG = "σ"
    STOP_TAG = "ε"
    EMBEDDING_DIM = 5
    HIDDEN_DIM = 4
    EPOCH = 10
    #####################################

    #Load training data
    training_data = reader.return_training()#[(list("ประเพณีการเทศน์มหาชาติ"), list("0000001001000010000001"))]
    
    #initialize look up tables 
    char2ix, ix2char = getDictionary()
    tag2ix = {"0": 0, "1": 1, START_TAG: 2, STOP_TAG: 3}

    #initialize model and optimizer TODO Adam optimizer 
    model = BiLSTM_CRF(char2ix, tag2ix, EMBEDDING_DIM, HIDDEN_DIM, tag2ix[START_TAG], tag2ix[STOP_TAG])
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

    # Make sure prepare_sequence from earlier in the LSTM section is loaded
    for epoch in range(
            EPOCH):  # again, normally you would NOT do 300 epochs, it is toy data
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

    checkPredictions(training_data, char2ix, tag2ix)

def main():
    # args = set(sys.argv)
    # if 'debug' in args:
    #     print("---------------------------")
    #     print("### Running Debug mode! ###")
    #     print("---------------------------")
    #     torch.manual_seed(1)

    # if 'train' in args:
    #     train()

    # if 'test' in args:
    #     test()
    #print(__doc__)
    args = docopt(__doc__)
    if args['train']:
        train(args)
    elif args['test']:
        test(args)
    else:
        raise RuntimeError('invalid run mode')


if __name__ == '__main__':
    main()