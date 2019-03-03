# -*- coding: utf-8 -*-
"""
run.py
-------------------
This file will initialize the dataset, character lookup table
and run training/tsting
"""
import sys
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

def train():
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
    for tgt, gold in zip_longest(args['OUTPUT_FILE'], args['TEST_GOLD_FILE']):
        print("dummy")

#Similar to the Decode function within assignment a5
def test():
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
    args = set(sys.argv)
    if 'debug' in args:
        print("---------------------------")
        print("### Running Debug mode! ###")
        print("---------------------------")
        torch.manual_seed(1)

    if 'train' in args:
        train()

    if 'test' in args:
        test()


if __name__ == '__main__':
    main()