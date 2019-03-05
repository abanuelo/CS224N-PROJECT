"""


Combined code from:

"""


import torch
import torch.nn as nn
import torch.nn.functional as F



class BiLSTM(nn.module):
	def __init__(self, char_vocab_size, word_vocab_size, num_tags):
		super(BiLSTM_CRF, self).__init__()
		self.embedding_dim = embedding_dim
		self.hidden_dim = hidden_dim

	def forward(self, sentence):
		pass



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
            'args': dict(char2ix=self.char2ix, tag2ix=self.tag2ix, embed_dim=self.embedding_dim, 
                            hidden_size=self.hidden_dim, start_id=self.start_id, stop_id=self.stop_id),
            'state_dict': self.state_dict()
        }

        torch.save(params, path)


class CRF(nn.Module):
	def __init__(self, num_tags, batch_size, start_id, stop_id, pad_id):
		super().__init__()
		self.num_tags = num_tags
		self.batch_size = batch_size
		self.start_id = start_id
		self.stop_id = stop_id
		self.pad_id = pad_id

		 # matrix of transition scores from j to i
        self.trans = nn.Parameter(randn(num_tags, num_tags))
        self.trans.data[start_id, :] = -10000. # no transition to SOS
        self.trans.data[:, stop_id] = -10000. # no transition from EOS except to PAD
        self.trans.data[:, pad_id] = -10000. # no transition from PAD except to PAD
        self.trans.data[pad_id, :] = -10000. # no transition to PAD except from EOS
        self.trans.data[pad_id, stop_id] = 0.
        self.trans.data[pad_id, pad_id] = 0.

    def forward(self, sentence):
    	#initialize alphas 
    	init_alphas = torch.full((self.batch_size, self.num_tags), -10000.)
    	score[:, SOS_IDX] = 0.


class BiLSTM_CRF(nn.Module):
	def __init__(self, char_vocab_size, word_vocab_size, num_tags):
		super(BiLSTM_CRF, self).__init__()
		

	def forward(self, sentence):
		pass