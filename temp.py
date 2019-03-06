"""


Combined code from:
https://pytorch.org/tutorials/beginner/nlp/advanced_tutorial.html
https://github.com/threelittlemonkeys/lstm-crf-pytorch
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

def log_sum_exp(x):
	m = torch.max(x, -1)[0]
	return m + torch.log(torch.sum(torch.exp(x - m.unsqueeze(-1)), -1))

class BiLSTM(nn.Module):
	def __init__(self, vocab_size, num_tag, embedding_dim, hidden_dim, pad_id):
		super(BiLSTM, self).__init__()
		self.embedding_dim = embedding_dim
		self.hidden_dim = hidden_dim
		self.pad_id = pad_id
		
		#character embeddings
		self.embeding = nn.Embedding(vocab_size, embedding_dim)  
		self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
							num_layers=1, bidirectional=True, batch_first = True)

		# Maps the output of the LSTM into tag space.
		self.hidden2tag = nn.Linear(hidden_dim, num_tag)


	def forward(self, inp, mask):
		"""
		@param inp(tensor) -- input of the 
		@param mask (tensor ) -- a tensor that is 1 if the character 
								is not padding 0 otherwise
		"""
		X = self.embeding(inp)
		lengths = mask.sum(1).int()
		X_packed = pack_padded_sequence(X, lengths, batch_first = True)
		hidden, _ = self.lstm(X_packed)
		hidden, _ = pad_packed_sequence(hidden, padding_value = self.pad_id, batch_first = True)
		h_out = self.hidden2tag(hidden) #(batch_size, max_sent_len, num_tag)
		h_out *= mask.unsqueeze(2)
		return h_out
		

	def decode():
		pass



class CRF(nn.Module):
	def __init__(self, num_tags, batch_size, start_id, stop_id, pad_id):
		super(CRF, self).__init__()
		self.num_tags = num_tags
		self.batch_size = batch_size
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

	def forward(self, h_tag, mask): #(batch_size, max_sent_len, tag_size)????
		#initialize alphas 
		score = torch.full((self.batch_size, self.num_tags), -10000., dtype=torch.float)
		score[:, self.stop_id] = 0. #set the stop score to 0
		#trans = self.trans.unsqueeze(0) #(1,num_tags,num_tags)

		# iterate over sentence (max_sent_len)
		for t in range(h_tag.size(1)):  
			mask_t = mask[:, t].unsqueeze(1) #get t'th mask (batch_size, 1, 1)
			emit_t = h_tag[:, t].unsqueeze(2) # (batch_size, num_tags, 1)
			score_t_trans = score.unsqueeze(1) + self.trans
			score_t = score_t_trans + emit_t
			score_t = log_sum_exp(score_t) # [batch_size, num_tags, num_tags] -> [batch_size, num_tags]
			score = score_t * mask_t + score * (1 - mask_t)
		score += self.trans[self.stop_id]
		score = log_sum_exp(score)
		return score


	def decode(self, h_tag, mask): #(batch_size, max_sent_len, tag_size)????
		#initialize alphas 
		bptr = torch.tensor([],dtype=torch.long)
		score = torch.full((self.batch_size, self.num_tags), -10000., dtype=torch.float)
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

		# print(bptr)
		# print(bptr.size())
		# print(best_tag)
		bptr = bptr.tolist()
		best_path = [[i] for i in best_tag.tolist()]
		print(best_path)
		for b in range(self.batch_size):
			x = best_tag[b] # best tag
			y = int(mask[b].sum().item()) #get length
			for bptr_t in reversed(bptr[b][:y]):
				x = bptr_t[x]
				best_path[b].append(x)
			best_path[b].pop() #pop the start token
			best_path[b].reverse()
		return best_path



	# def decode(self):
	# 	pass




class BiLSTM_CRF(nn.Module):
	def __init__(self, vocab_size, embedding_dim, hidden_dim, num_tags, start_id, stop_id, pad_id):
		super(BiLSTM_CRF, self).__init__()
		self.lstm = BiLSTM()
		self.crf = CRF()
		
	def forward(self, input):
		mask = 1-sents.data.eq(pad_id).float()
		h_out = self.lstm(input, mask)
		return crf(h_out, mask)

	# def decode(self):
	# 	pass


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