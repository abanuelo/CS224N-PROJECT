#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Usage:
    run.py train --train-input=<file> --train-gold=<file> [options]
    run.py test --test-input=<file> --test-gold=<file> [options]

Options:
    -d --debug                    Enable Debug mode
    --train-input=<file>          Training input path
    --train-gold=<file>           Training gold path
    --test-input=<file>           Testing input path
    --test-gold=<file>            Testing gold path
    --batch-size=<x>              Set Batch size
    --valid-niter=<x>             Set maximum iterations per epoch
    --max-epoch=<x>               Set maximum number of epochs
    --save-path=<file>            Sets the save path for the model
    --cuda                        Use the gpu

This file will initialize the dataset, character lookup table
and run training/tsting           

"""
import sys
from docopt import docopt
import torch
from itertools import zip_longest
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
    ##########DEFAULT PARAMETER SETTINGS########
    START_TAG = "σ"
    STOP_TAG = "ε"
    EMBEDDING_DIM = 5
    HIDDEN_DIM = 4
    EPOCH = 100
    #####################################

    if args['--max-epoch']:
        EPOCH = int(args['--max-epoch'])

    #Load training data
    training_data = reader.return_training(args['--train-input'], args['--train-gold'])
    
    #initialize look up tables 
    char2ix, ix2char = getDictionary()
    tag2ix = {"0": 0, "1": 1, START_TAG: 2, STOP_TAG: 3}

    #Allow Debug mode
    if args['--debug']:
        seed = 1234
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    #initialize model and optimizer TODO Adam optimizer 
    model = BiLSTM_CRF(char2ix, tag2ix, EMBEDDING_DIM, HIDDEN_DIM, tag2ix[START_TAG], tag2ix[STOP_TAG])
    optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

    #Set device
    device = torch.device("cuda:0" if args['--cuda'] else "cpu")
    print('Use device: %s' % device, file=sys.stderr)
    model = model.to(device)

    # Make sure prepare_sequence from earlier in the LSTM section is loaded

    for epoch in range(EPOCH):  # again, normally you would NOT do 300 epochs, it is toy data
        epoch_loss = 0.
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
            epoch_loss += float(loss[0])
            loss.backward()
            optimizer.step()
        print('epoch: ' +str(epoch)+' loss: ' +str(epoch_loss))

    checkPredictions(training_data, char2ix, tag2ix)

def write_to_output(test_data_tgt, char2ix, tag2ix):
    """ Computes the F1 Score reported by the trained model while writing 
    to the necessary output file for testing purposes
    """
    prepared_test_data_tgt = []

    with open(args['OUTPUT_FILE'], 'w') as f:
        for sent in test_data_tgt:
            char_seq = prepare_sequence(sent, char2ix)
            prepared_test_data_tgt.append(char_seq)
            tokenized_output = ''.join(model(prepared_test_data_tgt))
            f.write(tokenized_output + '\n')

def compute_F1_scores():
    F1 = 0
    for tgt, gold in zip_longest(args['OUTPUT_FILE'], args['TEST_GOLD_FILE']):
        characters = list(tgt.strip('\n'))
        characters_gold = list(gold.strip('\n'))
        TP, FP, FN = 0
        for i in range(len(characters)):
            if characters[i] == characters_gold[j] and characters[i] == '1':
                TP += 1
            elif characters[i] != characters_gold[j] and characters[i] == '0':
                FP += 1 
            elif characters[i] != characters_gold[j] and characters[i] == '1':
                FN += 1
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        F1 = (2*precision*recall) / (precision+recall)
    return F1


#Similar to the Decode function within assignment a5
def test(args:dict):
    print("load test source input from [{}]".format(args['TEST_INPUT_FILE']), file=sys.stderr)
    test_data_src = reader.read_corpus(args['TEST_INPUT_FILE'], source='src')
    if args['TEST_GOLD_FILE']:
        print("load test target output from [{}]".format(args['TEST_GOLD_FILE']), file=sys.stderr)
        test_data_tgt = reader.read_corpus(args['TEST_GOLD_FILE'], source='tgt')

    print("load model from {}".format(args['MODEL_PATH']), file=sys.stderr)
    model = BiLSTM_CRF.load(args['MODEL_PATH'])

    if args['--cuda']:
        model = model.to(torch.device("cuda:0"))

    if args['TEST_TARGET_FILE']:
        write_to_output(test_data_tgt, char2ix, tag2ix)

    F1 = compute_F1_scores()
    print("F1 Score: {}".format(F1))


def main():
    args = docopt(__doc__)
    if args['train']:
        train(args)
    elif args['test']:
        test(args)
    else:
        raise RuntimeError('invalid run mode')


if __name__ == '__main__':
    main()