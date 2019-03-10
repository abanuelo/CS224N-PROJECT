"""


Combined code from:
https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html
https://github.com/threelittlemonkeys/lstm-crf-pytorch
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

def log_sum_exp(x):
    m = torch.max(x, -1)[0]
    return m + torch.log(torch.sum(torch.exp(x - m.unsqueeze(-1)), -1))
    # max_score = x[0, argmax(vec)]
    # max_score_broadcast = max_score.view(1, -1).expand(1, x.size()[1])
    # return max_score + torch.log(torch.sum(torch.exp(x - max_score_broadcast)))

# class Embedding(nn.Module):
#   def __init__(self, embedding_dim, vocab_size):
#       super(Embedding, self).__init__()
#       self.embeding = nn.Embedding(vocab_size, embedding_dim)

#   def forward(inp):
#       return self.embeding(inp)


class BiLSTM(nn.Module):
    def __init__(self, num_tag, embedding_dim, hidden_dim, pad_id):
        super(BiLSTM, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.pad_id = pad_id
        
        #character embeddings
        self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
                            num_layers=1, bidirectional=True, batch_first = True)

        # Maps the output of the LSTM into tag space.
        self.hidden2tag = nn.Linear(hidden_dim, num_tag)


    def forward(self, inp_embed, mask):
        """
        @param inp(tensor) -- input of the 
        @param mask (tensor ) -- a tensor that is 1 if the character 
                                is not padding 0 otherwise
        """ 
        lengths = mask.sum(1).int()
        X_packed = pack_padded_sequence(inp_embed, lengths, batch_first = True)
        hidden, _ = self.lstm(X_packed)
        hidden, _ = pad_packed_sequence(hidden, padding_value = self.pad_id, batch_first = True)
        h_tag = self.hidden2tag(hidden) #(batch_size, max_sent_len, num_tag)
        h_tag *= mask.unsqueeze(2)
        return h_tag




class CRF(nn.Module):
    def __init__(self, num_tags, start_id, stop_id, pad_id):
        super(CRF, self).__init__()
        self.num_tags = num_tags
        self.start_id = start_id
        self.stop_id = stop_id
        self.pad_id = pad_id

         # matrix of transition scores from j to i
        self.trans = nn.Parameter(torch.randn(num_tags, num_tags))
        self.trans.data[start_id, :] = -10000. # no transition to start
        self.trans.data[:, stop_id] = -10000. # no transition from end except to padding
        self.trans.data[:, pad_id] = -10000. # no transition from pad except to pading
        self.trans.data[pad_id, :] = -10000. # no transition to padding except from end
        self.trans.data[pad_id, stop_id] = 0. # stop to pad
        self.trans.data[pad_id, pad_id] = 0. #pad to pad

    def forward(self, h_tag, mask): #(batch_size, max_sent_len, tag_size)
        #initialize alphas 
        batch_size = len(h_tag)
        score = torch.full((batch_size, self.num_tags), -10000., dtype=torch.float, device=self.trans.data.device)
        score[:, self.stop_id] = 0. #set the stop score to 0
        trans = self.trans.unsqueeze(0) #(1,num_tags,num_tags)
        # iterate over sentence (max_sent_len)
        for t in range(h_tag.size(1)):  
            mask_t = mask[:, t].unsqueeze(1) #get t'th mask (batch_size, 1, 1)
            emit_t = h_tag[:, t].unsqueeze(2) # (batch_size, num_tags, 1)
            score_t = score.unsqueeze(1) + emit_t + trans 
                            # (batch_size, num_tags, num_tags) -> [batch_size, num_tags, num_tags]
            score_t = log_sum_exp(score_t) # [batch_size, num_tags, num_tags] -> [batch_size, num_tags]
            score = score_t * mask_t + score * (1 - mask_t) #[batch_size, num_tags]
        score += self.trans[self.stop_id]
        score = log_sum_exp(score)
        return score


    def decode(self, h_tag, mask): #(batch_size, max_sent_len, tag_size)????
        #initialize alphas 
        print(mask)
        batch_size = len(h_tag)
        bptr = torch.tensor([],dtype=torch.long, device=self.trans.data.device)
        score = torch.full((batch_size, self.num_tags), -10000., dtype=torch.float, device=self.trans.data.device)
        score[:, self.stop_id] = 0. #set the stop score to 0
        #trans = self.trans.unsqueeze(0) #(1,num_tags,num_tags)

        # iterate over sentence (max_sent_len)
        for t in range(h_tag.size(1)):  
            mask_t = mask[:, t].unsqueeze(1) #get t'th mask (batch_size, 1, 1)
            emit_t = h_tag[:, t].unsqueeze(2) # (batch_size, num_tags, 1)
            score_t_trans = score.unsqueeze(1) + self.trans
            score_x, bptr_t = score_t_trans.max(2)
            bptr = torch.cat((bptr, bptr_t.unsqueeze(1)), 1)
            score_t = score_t_trans + emit_t
            score_t = log_sum_exp(score_t) # [batch_size, num_tags, num_tags] -> [batch_size, num_tags]
            score = score_t * mask_t + score * (1 - mask_t)
        score += self.trans[self.stop_id]
        best_score, best_tag = score.max(1)
        score = log_sum_exp(score)

        bptr = bptr.tolist()
        print(bptr)
        best_path = [[i] for i in best_tag.tolist()]
        for b in range(batch_size):
            x = best_tag[b] # best tag
            y = int(mask[b].sum().item()) #get length
            for bptr_t in reversed(bptr[b][:y]):
                x = bptr_t[x]
                best_path[b].append(x)
            best_path[b].pop() #pop the start token
            best_path[b].reverse()
        return best_path


    def score(self, h_tag, gold, mask): # calculate the score of a given sequence
        batch_size = len(h_tag)
        score = torch.full((batch_size,), 0., dtype=torch.float, device=self.trans.data.device)
        h_tag = h_tag.unsqueeze(3)
        trans = self.trans.unsqueeze(2)
        for t in range(h_tag.size(1)-1): # iterate through except the last element
            mask_t = mask[:, t]
            emit_t = torch.cat([h_tag[t, gold[t + 1]] for h_tag, gold in zip(h_tag, gold)])
            trans_t = torch.cat([trans[gold[t + 1], gold[t]] for gold in gold])
            score += (emit_t + trans_t) * mask_t
        last_tag = gold.gather(1, mask.sum(1).long().unsqueeze(1)-1).squeeze(1)
        score += self.trans[self.stop_id, last_tag]
        return score



class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, num_tag, embedding_dim, hidden_dim, start_id, stop_id, pad_id):
        super(BiLSTM_CRF, self).__init__()
        #parameters
        self.vocab_size = vocab_size
        self.num_tag = num_tag
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.start_id = start_id
        self.stop_id = stop_id
        self.pad_id = pad_id

        #Models
        self.embeding = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = BiLSTM(num_tag, embedding_dim, hidden_dim, pad_id)
        self.crf = CRF(num_tag, start_id, stop_id, pad_id)


    def forward(self, inp, gold, mask): # for training
        inp_embed = self.embeding(inp)
        h_tag = self.lstm(inp_embed, mask)
        Z = self.crf.forward(h_tag, mask)
        score = self.crf.score(h_tag, gold, mask)
        return Z - score # NLL loss
        
    def decode(self, inp, mask):
        inp_embed = self.embeding(inp)
        h_out = self.lstm(inp_embed, mask)
        return self.crf.decode(h_out, mask)

    @staticmethod
    def load(model_path: str):
        """ Load the model from a file.
        @param model_path (str): path to model
        """
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        model = BiLSTM_CRF(**args)
        model.load_state_dict(params['state_dict'])
        return model


    def save(self, path: str):
        """ Save the odel to a file.
        @param path (str): path to the model
        """
        print('save model parameters to [%s]' % path, file=sys.stderr)
        params = {
            'args': dict(vocab_size=self.vocab_size, num_tag=self.num_tag, embedding_dim=self.embedding_dim, 
                hidden_dim=self.hidden_dim, start_id=self.start_id, stop_id=self.stop_id, pad_id=self.pad_id),
            'state_dict': self.state_dict()
        }

        torch.save(params, path)