import os
import math
import numpy as np
import torch
from itertools import zip_longest
import torch.autograd as autograd
import torch.nn as nn
import torch.optim as optim

def get_data(input_path, gold_path):
    data = []
    #Reading in Data from the Train Set
    with open(input_path) as textfile1, open(gold_path) as textfile2: 
        for x, y in zip_longest(textfile1, textfile2):
            characters = list(x.strip('\n'))
            gold = list(y.strip('\n'))
            tuple_char = (characters, gold)
            data.append(tuple_char)

    return data


def batch_iter(data, batch_size:int, shuffle=False):
    """ Yield batches of source and target sentences reverse sorted by length (largest to smallest).
    @param data tuple: list of tuples containing source and target sentence
    @param batch_size (int): batch size
    @param shuffle (boolean): whether to randomly shuffle the dataset

    modified from assignment 4, CS224N starter code
    """
    batch_num = math.ceil(len(data) / batch_size)
    index_array = list(range(len(data)))

    if shuffle:
        np.random.shuffle(index_array)

    for i in range(batch_num):
        indices = index_array[i * batch_size: (i + 1) * batch_size]
        examples = [data[idx] for idx in indices]
        examples = sorted(examples, key=lambda e: len(e[0]), reverse=True)
        characters = [e[0] for e in examples]
        gold = [e[1] for e in examples]
        yield characters, gold


def pad_ids(ids:list, pad_id):
    """
    Pads an id list to the longest sentence

    @param ids a list of lists of ids 
    @param pad_id a list of ids that is padded 
    """
    max_len = len(ids[0])
    sents_padded = []

    for sent in ids:
        sents_padded.append(sent + [pad_id]*(max_len-len(sent)))
    return sents_padded 

def sents2tensor(sents: list, char2ix:dict, pad_id:int, device: torch.device):
    
    """
    Creates a padded tensor from a list of sentences

    @param sents -- a list of strings
    @param char2ix -- a dictionary from character to id
    @param pad_id -- id of padtoken
    @param device -- device of the model

    @return a tensor of dim (max_string_length, batch_size)
    """
    
    ids = [[char2ix[c] for c in s] for s in sents]
    padded = pad_ids(ids, pad_id)
    data = torch.tensor(padded, dtype=torch.long, device=device)
    return data

# def sents2packed(sents:list, to_ix:dict, device: torch.device):
#   """
#   Creates a packed tensor from a list of sentences

#   @param sents -- a list of strings (assumes it is ordered by length)
#   @param char2ix -- a dictionary from character to id
#   @param device -- device of the model

#   @return a tensor of dim (max_string_length, batch_size)
#   """
#   ids = [torch.tensor([to_ix[c] for c in s], dtype=torch.long, device=device) for s in sents]
#   return nn.utils.rnn.pack_sequence(ids)





