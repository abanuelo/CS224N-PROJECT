#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Usage:
    run.py train --train-input=<file> --train-gold=<file> --dev-input=<file> --dev-gold=<file>[options]
    run.py test --test-input=<file> --test-gold=<file> --model-output=<file> [options]


Options:
    -d --debug                       Enable Debug mode
    --train-input=<file>             Training input path
    --train-gold=<file>              Training gold path
    --test-input=<file>              Testing input path
    --test-gold=<file>               Testing gold path
    --dev-input=<file>               Dev input path
    --dev-gold=<file>                Dev gold path
    --model-output=<file>            Model output path
    --seed=<int>                     seed  [default: 0]
    --batch-size=<int>               batch size  [default: 32]
    --embed-size=<int>               embedding size  [default: 64]
    --hidden-size=<int>              hidden size  [default: 128]
    --clip-grad=<float>              gradient clipping  [default: 0.1]
    --log-every=<int>                log every  [default: 10]
    --max-epoch=<int>                max epoch  [default: 30]
    --input-feed                     use input feeding
    --patience=<int>                 wait for how many iterations to decay learning rate  [default: 5]
    --max-num-trial=<int>            terminate training after how many trials  [default: 5]
    --lr-decay=<float>               learning rate decay  [default: 0.001]
    --beam-size=<int>                beam size  [default: 5]
    --sample-size=<int>              sample size  [default: 5]
    --lr=<float>                     learning rate [default: 0.001]
    --uniform-init=<float>           uniformly initialize all parameters  [default: 0.1]
    --save-to=<file>                 model save path [default: model.bin]
    --valid-niter=<int>              perform validation after how many iterations [default: 2000]
    --dropout=<float>                dropout  [default: 0.3]
    --max-decoding-time-step=<int>   maximum number of decoding time steps  [default: 70]
    --cuda                           Use the gpu

This file will initialize the dataset, character lookup table
and run training/tsting        


# includes modified code from assignment 4 & 5, CS224N (Winter 2019) Stanford university
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
import numpy as np
from f1 import compute_F1_scores
#import reader
from  BiLSTM_CRF import BiLSTM_CRF
from utils import batch_iter, get_data, sents2tensor

#####################################################################
# Run training

class Run():
    def __init__(self):
        self.start_tag = "σ"
        self.stop_tag = "ε"
        self.padding = "π"
        self.target_padding = 4
        self.args = docopt(__doc__)
        self.char2id, self.id2char = self.get_dictionary()
        self.tag2id = {"0": 0, "1": 1, self.start_tag: 2, self.stop_tag: 3, self.padding: 4}
        self.id2tag = {value:key for key,value in  self.tag2id.items()}
        self.device = torch.device("cuda:0" if self.args['--cuda'] else "cpu")
        self.model = None
        self.optimizer = None

    def get_dictionary(self):
        """
        Creates a lookup table for all characters
        Greek characters are used as special symbols
            (abbreviations, named entities, start, and stop)
        """
        extra_chars = "αβσεπ"
        thai_chars = "กขฃคฅฆงจฉชซฌญฎฏฐฑฒณดตถทธนบปผฝพฟภมยรฤลฦวศษสหฬอฮฯะั าำิ ี ึ ื ุ ู ฺ฿เแโใไๅๆ็ ่ ้ ๊ ๋ ์ ํ ๎๐๑๒๓๔๕๖๗๘๙".replace(" ", "")
        eng_chars = " !\"#$%&\'()*+,-./0123456789:;<=>?@ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`abcdefghijklmnopqrstuvwxyz{|}‘’~"
        all_chars = extra_chars+thai_chars+eng_chars
        char2id = {c:i for i,c in enumerate(list(all_chars))}
        id2char = {i:c for i,c in enumerate(list(all_chars))}
        return char2id, id2char

    def evaluate_ppl(self, dev_data, batch_size:int =32):  #batch_size=32):
        """ Evaluate perplexity on dev sentences

        @param dev_data (list of (src_sent, gold_sent)): list of tuples containing source and target sentence
        @param batch_size (int): batchsize of dev set

        @returns ppl (perplixty on dev sentences)
        """
        was_training = self.model.training
        self.model.eval()

        cum_loss = 0.
        cum_gold_chars = 0.

        # no_grad() signals backend to throw away all gradients
        with torch.no_grad():
            for inp, gold in batch_iter(dev_data, batch_size):
                inp_tensor = sents2tensor(inp, self.char2id, self.char2id[self.padding], self.device)
                gold_tensor = sents2tensor(gold, self.tag2id, self.target_padding, self.device)
                mask = 1-inp_tensor.data.eq(self.char2id[self.padding]).float()
                loss = -self.model(inp_tensor, gold_tensor, mask).sum()

                cum_loss += loss.item()
                #print("cum_loss", cum_loss)
                gold_char_num_to_predict = sum(len(s[1:]) for s in gold)  # omitting leading `<s>`
                cum_gold_chars += gold_char_num_to_predict
                #print("cum_gold_chars", cum_gold_chars)

            ppl = np.exp(cum_loss / cum_gold_chars)

        if was_training:
            self.model.train()
        return ppl


    def checkPredictions(self,training_data):
        # Check predictions before training
        with torch.no_grad():
            precheck_sent = prepare_sequence(training_data[0][0], self.char2id)
            print(self.model(precheck_sent))

    def train(self):
        embedding_dim = int(self.args['--embed-size'])
        hidden_dim = int(self.args['--hidden-size'])
        epoch = int(self.args['--max-epoch'])
        train_batch_size = int(self.args['--batch-size'])
        clip_grad = float(self.args['--clip-grad'])
        log_every = int(self.args['--log-every'])
        valid_niter = int(self.args['--valid-niter'])
        model_save_path = self.args['--save-to']
        dev_data = get_data(self.args['--dev-input'], self.args['--dev-gold'])

        #Load training data
        training_data = get_data(self.args['--train-input'], self.args['--train-gold'])
                
        #Allow Debug mode
        if self.args['--debug']:
            seed = 1234
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

        #initialize model
        self.model = BiLSTM_CRF(len(self.char2id), len(self.tag2id), embedding_dim,
                     hidden_dim, self.tag2id[self.start_tag], self.tag2id[self.stop_tag], self.char2id[self.padding])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=float(self.args['--lr']))

        #Set self.device
        print('Use self.device: %s' % self.device, file=sys.stderr)
        self.model = self.model.to(self.device)

        #initialize variables for training
        num_trial = 0
        train_iter = patience = cum_loss = report_loss = report_gold_chars = cum_gold_chars = 0
        cum_examples = report_examples = valid_num = 0
        hist_valid_scores = []
        train_time = begin_time = time.time()



        #####  BEGIN TRAINING   ########################################################################


        for e in range(epoch):  # again, normally you would NOT do 300 epochs, it is toy data
            for sentence, tags in batch_iter(training_data, train_batch_size, shuffle=False):
                train_iter += 1
                #Step 1:
                self.optimizer.zero_grad()
                #self.model.zero_grad()

                batch_size = len(sentence) #This might be different from train_batch_size in the last iteration

                #Step 2
                sentence_in = sents2tensor(sentence, self.char2id, self.char2id[self.padding], self.device)
                targets = sents2tensor(tags, self.tag2id, self.target_padding, self.device)
                #targets = torch.tensor([[self.tag2id[c] for c in t] for t in tags], dtype=torch.long, self.device=self.device)

                # Step 3. Run our forward pass.
                mask = 1-sentence_in.data.eq(self.char2id[self.padding]).float()
                loss = torch.mean(self.model(sentence_in, targets, mask))
                batch_loss = loss#loss.sum()

                # Step 4. Compute the loss, gradients, and update the parameters by
                # calling self.optimizer.step()
                #epoch_loss += float(loss[0])
                loss.backward()

                #clip gradinet
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), clip_grad)
                self.optimizer.step()

                batch_losses_val = batch_loss.item()
                report_loss += batch_losses_val
                cum_loss += batch_losses_val
                report_examples += batch_size
                gold_chars_num_to_predict = sum(len(c) for c in tags)
                cum_gold_chars += gold_chars_num_to_predict
                report_gold_chars += gold_chars_num_to_predict
                cum_examples += batch_size

                if train_iter % log_every == 0:
                    print('epoch %d, iter %d, avg. loss %.2f, avg. ppl %.2f ' \
                          'cum. examples %d, speed %.2f chars/sec, time elapsed %.2f sec' % (e, train_iter,
                                                                                             report_loss / report_examples,
                                                                                             loss,
                                                                                             cum_examples,
                                                                                             report_gold_chars / (time.time() - train_time),
                                                                                             time.time() - begin_time), file=sys.stderr)

                    train_time = time.time()
                    report_loss = report_gold_chars = report_examples = 0.

                #print("cum_loss", cum_loss)
                #print("cum_gold_chars", cum_gold_chars)
                if train_iter % valid_niter == 0:
                    print('epoch %d, iter %d, cum. loss %.2f, cum. ppl %.2f cum. examples %d' % (e, train_iter,
                                                                             cum_loss / cum_examples,
                                                                             np.exp(cum_loss / cum_gold_chars),
                                                                             cum_examples), file=sys.stderr)

                    cum_loss = cum_examples = cum_gold_chars = 0.
                    valid_num += 1

                    print('begin validation ...', file=sys.stderr)
                    dev_ppl = self.evaluate_ppl(dev_data, batch_size=128)
                    valid_metric = -dev_ppl

                    print('validation: iter %d, dev. ppl %f' % (train_iter, dev_ppl), file=sys.stderr)

                    is_better = len(hist_valid_scores) == 0 or valid_metric > max(hist_valid_scores)
                    hist_valid_scores.append(valid_metric)

                    if is_better:
                        patience = 0
                        print('save currently the best model to [%s]' % model_save_path, file=sys.stderr)
                        self.model.save(model_save_path)

                        # also save the optimizers' state
                        torch.save(self.optimizer.state_dict(), model_save_path + '.optim')
                    elif patience < int(self.args['--patience']):
                        patience += 1
                        print('hit patience %d' % patience, file=sys.stderr)

                        if patience == int(self.args['--patience']):
                            num_trial += 1
                            print('hit #%d trial' % num_trial, file=sys.stderr)
                            if num_trial == int(self.args['--max-num-trial']):
                                print('early stop!', file=sys.stderr)
                                exit(0)

                            # decay lr, and restore from previously best checkpoint
                            lr = self.optimizer.param_groups[0]['lr'] * float(self.args['--lr-decay'])
                            print('load previously best model and decay learning rate to %f' % lr, file=sys.stderr)

                            # load model
                            params = torch.load(model_save_path, map_location=lambda storage, loc: storage)
                            self.model.load_state_dict(params['state_dict'])
                            self.model = self.model.to(self.device)

                            print('restore parameters of the optimizers', file=sys.stderr)
                            self.optimizer.load_state_dict(torch.load(model_save_path + '.optim'))

                            # set new lr
                            for param_group in self.optimizer.param_groups:
                                param_group['lr'] = lr

                            # reset patience
                            patience = 0



        print('reached maximum number of epochs!', file=sys.stderr)


    def write_to_output(self, output_gold, file):
        """ Computes the F1 Score reported by the trained model while writing 
        to the necessary output file for testing purposes
        """
        prepared_test_data_gold = []

        
        for sent in output_gold:
            line = ''.join([self.id2tag[c] for c in sent]) + '\n'
            file.write(line)


    #Similar to the Decode function within assignment a5
    def test(self):
        print("load test source input from [{}]".format(self.args['--test-input']), file=sys.stderr)    
        print("load test target output from [{}]".format(self.args['--test-gold']), file=sys.stderr)
        testing_data = get_data(self.args['--test-input'], self.args['--test-gold'])

        print("load model from {}".format(self.args['--save-to']), file=sys.stderr)
        model = BiLSTM_CRF.load(self.args['--save-to'])

        if self.args['--cuda']:
            self.model = self.model.to(torch.self.device("cuda:0"))

        with open(self.args['--model-output'], 'w+') as f:
            for inp, _ in batch_iter(testing_data, batch_size=1):
                inp_tensor=sents2tensor(inp, self.char2id, self.char2id[self.padding], self.device)
                mask = 1-inp_tensor.data.eq(self.char2id[self.padding]).float()
                model_output = model.decode(inp_tensor, mask)
                self.write_to_output(model_output, f)

        F1_micro, F1_macro = compute_F1_scores(self.args['--model-output'], self.args['--test-gold'])
        print("F1_micro", F1_micro)
        print("F1_macro", F1_macro)


def main():
    args = docopt(__doc__)
    run = Run()
    if args['train']:
        run.train()
    elif args['test']:
        run.test()
    else:
        raise RuntimeError('invalid run mode')


if __name__ == '__main__':
    main()