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
import math
import time
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

def evaluate_ppl(model, dev_data, batch_size=32):
    """ Evaluate perplexity on dev sentences
    @param model (NMT): NMT Model
    @param dev_data (list of (src_sent, tgt_sent)): list of tuples containing source and target sentence
    @param batch_size (batch size)
    @returns ppl (perplixty on dev sentences)
    """
    was_training = model.training
    model.eval()

    cum_loss = 0.
    cum_tgt_words = 0.

    # no_grad() signals backend to throw away all gradients
    with torch.no_grad():
        for src_sents, tgt_sents in batch_iter(dev_data, batch_size):
            loss = -model(src_sents, tgt_sents).sum()

            cum_loss += loss.item()
            tgt_word_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting leading `<s>`
            cum_tgt_words += tgt_word_num_to_predict

        ppl = np.exp(cum_loss / cum_tgt_words)

    if was_training:
        model.train()

    return ppl


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

    train_batch_size = int(args['--batch-size'])
    clip_grad = float(args['--clip-grad'])
    log_every = int(args['--log-every'])
    valid_niter = int(args['--valid-niter'])

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
    optimizer = torch.optim.Adam(model.parameters(), lr=float(args['--lr']))

    #Set device
    device = torch.device("cuda:0" if args['--cuda'] else "cpu")
    print('Use device: %s' % device, file=sys.stderr)
    model = model.to(device)

    #initialize variables for training
    num_trial = 0
    train_iter = patience = cum_loss = report_loss = cum_tgt_chars = report_tgt_chars = 0
    cum_examples = report_examples = epoch = valid_num = 0
    hist_valid_scores = []
    train_time = begin_time = time.time()

    # Make sure prepare_sequence from earlier in the LSTM section is loaded

    while True:
        epoch += 1
        for train_input, train_gold in batch_iter(training_data, batch_size = train_batch_size, shuffle=False):
            train_iter += 1
            optimizer.zero_grad()
            batch_size = len(train_input)
            example_losses = -model(train_input, train_gold)
            batch_loss = example_losses.sum()
            loss = batch_loss / batch_size 
            loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            optimizer.step()
            batch_losses_val = batch_loss.item()
            report_loss += batch_losses_val
            cum_loss += batch_losses_val
            tgt_chars_num_to_predict = sum(len(s[1:]) for s in train_gold)  # omitting leading `<s>`
            report_tgt_chars += tgt_chars_num_to_predict
            cum_tgt_chars += tgt_chars_num_to_predict
            report_examples += batch_size
            cum_examples += batch_size

            if train_iter % log_every == 0:
                print('epoch %d, iter %d, avg. loss %.2f, avg. ppl %.2f ' \
                      'cum. examples %d, speed %.2f words/sec, time elapsed %.2f sec' % (epoch, train_iter,
                                                                                         report_loss / report_examples,
                                                                                         math.exp(report_loss / report_tgt_chars),
                                                                                         cum_examples,
                                                                                         report_tgt_chars / (time.time() - train_time),
                                                                                         time.time() - begin_time), file=sys.stderr)

                train_time = time.time()
                report_loss = report_tgt_chars = report_examples = 0.

            if train_iter % valid_niter == 0:
                print('epoch %d, iter %d, cum. loss %.2f, cum. ppl %.2f cum. examples %d' % (epoch, train_iter,
                                                                                         cum_loss / cum_examples,
                                                                                         np.exp(cum_loss / cum_tgt_words),
                                                                                         cum_examples), file=sys.stderr)

                cum_loss = cum_examples = cum_tgt_words = 0.
                valid_num += 1

                #print('begin validation ...', file=sys.stderr)

                # compute dev. ppl and bleu
                dev_ppl = evaluate_ppl(model, dev_data, batch_size=128)   # dev batch size can be a bit larger
                valid_metric = -dev_ppl

                print('validation: iter %d, dev. ppl %f' % (train_iter, dev_ppl), file=sys.stderr)

                is_better = len(hist_valid_scores) == 0 or valid_metric > max(hist_valid_scores)
                hist_valid_scores.append(valid_metric)

                if is_better:
                    patience = 0
                    print('save currently the best model to [%s]' % model_save_path, file=sys.stderr)
                    model.save(model_save_path)

                    # also save the optimizers' state
                    torch.save(optimizer.state_dict(), model_save_path + '.optim')
                elif patience < int(args['--patience']):
                    patience += 1
                    print('hit patience %d' % patience, file=sys.stderr)

                    if patience == int(args['--patience']):
                        num_trial += 1
                        print('hit #%d trial' % num_trial, file=sys.stderr)
                        if num_trial == int(args['--max-num-trial']):
                            print('early stop!', file=sys.stderr)
                            exit(0)

                        # decay lr, and restore from previously best checkpoint
                        lr = optimizer.param_groups[0]['lr'] * float(args['--lr-decay'])
                        print('load previously best model and decay learning rate to %f' % lr, file=sys.stderr)

                        # load model
                        params = torch.load(model_save_path, map_location=lambda storage, loc: storage)
                        model.load_state_dict(params['state_dict'])
                        model = model.to(device)

                        print('restore parameters of the optimizers', file=sys.stderr)
                        optimizer.load_state_dict(torch.load(model_save_path + '.optim'))

                        # set new lr
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr

                        # reset patience
                        patience = 0

            if epoch == int(args['--max-epoch']):
                print('reached maximum number of epochs!', file=sys.stderr)
                exit(0)


    # for epoch in range(EPOCH):  # again, normally you would NOT do 300 epochs, it is toy data
    #     epoch_loss = 0.
    #     for sentence, tags in training_data:
    #         # Step 1. Remember that Pytorch accumulates gradients.
    #         # We need to clear them out before each instance
    #         model.zero_grad()

    #         # Step 2. Get our inputs ready for the network, that is,
    #         # turn them into Tensors of word indices.
    #         sentence_in = prepare_sequence(sentence, char2ix)
    #         targets = torch.tensor([tag2ix[t] for t in tags], dtype=torch.long)

    #         # Step 3. Run our forward pass.
    #         loss = model.neg_log_likelihood(sentence_in, targets)

    #         # Step 4. Compute the loss, gradients, and update the parameters by
    #         # calling optimizer.step()
    #         epoch_loss += float(loss[0])
    #         loss.backward()
    #         optimizer.step()
    #     print('epoch: ' +str(epoch)+' loss: ' +str(epoch_loss))

    # checkPredictions(training_data, char2ix, tag2ix)

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
    F1_micro = 0
    F1_macro = 0
    F1_arr = []
    TP_macro, FP_macro, FN_macro = 0
    for tgt, gold in zip_longest(args['OUTPUT_FILE'], args['TEST_GOLD_FILE']):
        characters = list(tgt.strip('\n'))
        characters_gold = list(gold.strip('\n'))
        TP, FP, FN = 0
        for i in range(len(characters)):
            if characters[i] == characters_gold[j] and characters[i] == '1':
                TP += 1
                TP_macro += 1
            elif characters[i] != characters_gold[j] and characters[i] == '0':
                FP += 1 
                FP_macro += 1
            elif characters[i] != characters_gold[j] and characters[i] == '1':
                FN += 1
                FN_macro += 1
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        F1 = (2*precision*recall) / (precision+recall)
        F1_arr.append(F1)
    F1_micro = sum(F1_arr) / len(F1_arr)
    precision_macro = TP_macro / (TP_macro + FP_macro)
    recall_macro = TP_macro / (TP_macro + FN_macro)
    F1_macro = (2*precision_macro*recall_macro) / (precision_macro+recall_macro)
    return F1_micro, F1_macro


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

    F1_micro, F1_macro = compute_F1_scores()
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