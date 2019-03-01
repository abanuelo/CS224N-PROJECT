import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd

torch.manual_seed(1)


'''Helper Functions for this BiLSTM_CRF'''
def argmax(vec):
	# return the argmax as a python int
	_, idx = torch.max(vec, 1)
	return idx.item()


def prepare_sequence(seq, to_ix):
	idxs = [to_ix[w] for w in seq]
	return torch.tensor(idxs, dtype=torch.long)


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec):
	max_score = vec[0, argmax(vec)]
	max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
	return max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast)))



class BiLSTM_CRF(nn.Module):
	def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim):
		super(BiLSTM_CRF, self).__init__()
		self.embedding_dim = embedding_dim
		self.hidden_dim = hidden_dim
		self.vocab_size = vocab_size
		self.tag_to_ix = tag_to_ix
		self.tagset_size = len(tag_to_ix)

		#Initialize Bi-Directional LSTM	
		self.char_embeds = nn.Embedding(vocab_size, embedding_dim)
		self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2, num_layers=1, bidirectional=True)

		#Mapping output to tag space [0,1] where 0 is a nonexistent parse and 1 is a true parse
		self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

		#Score of transitions (ie whether a certain parsed character should be mapped to 0 or 1)
		self.transitions = nn.Parameter(torch.randn(self.tagset_size, self.tagset_size))

		#Constrained by start and stop characters
		self.transitions.data[tag_to_ix[START_TAG], :] = -10000 # float("-inf")
		self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

		self.hidden = (torch.randn(2, 1, self.hidden_dim // 2), torch.randn(2, 1, self.hidden_dim // 2))

		#initializes randomized hidden weights to be used for lstm feature extraction
		def init_hidden(self):
			return (torch.randn(2, 1, self.hidden_dim // 2), torch.randn(2, 1, self.hidden_dim // 2))

		#Grabs LSTM Features used for emission parameter in CRF calculation to estimate whether to parse or not
		def _get_lstm_features(self, characters):
			self.hidden = self.init_hidden()
			e_char = self.char_embeds(characters).view(len(characters), 1, -1) 
			lstm_out, self.hidden = self.lstm(e_char, self.hidden)
			lstm_out = lstm_out.view(len(characters), self.hidden_dim)
			lstm_feats = self.hidden2tag(lstm_out)
			return lstm_feats

		def forward(self, characters):
			#Retrieve Emission Scores from the BiDirectional LSTM
			lstm_feats = self._get_lstm_features(characters)
			# Find the best path, given the features.
			score, tag_seq = self._viterbi_decode(lstm_feats)
			return score, tag_seq

		def _viterbi_decode(self, feats):
			backpointers = []

			# Initialize the viterbi variables in log space
			init_vvars = torch.full((1, self.tagset_size), -10000.)
			init_vvars[0][self.tag_to_ix[START_TAG]] = 0
			forward_var = init_vvars

			# forward_var at step i holds the viterbi variables for step i-1
			for feat in feats:
				bptrs_t = []  # holds the backpointers for this step
				viterbivars_t = []  # holds the vit

				for next_tag in range(self.tagset_size):
					# next_tag_var[i] holds the viterbi variable for tag i at the
					# previous step, plus the score of transitioning
					# from tag i to next_tag.
					# We don't include the emission scores here because the max
					# does not depend on them (we add them in below)
					next_tag_var = forward_var + self.transitions[next_tag]
					best_tag_id = argmax(next_tag_var)
					bptrs_t.append(best_tag_id)
					viterbivars_t.append(next_tag_var[0][best_tag_id].view(1)) #check dimesions during run
				# Now add in the emission scores, and assign forward_var to the set
		   		# of viterbi variables we just computed
				forward_var = (torch.cat(viterbivars_t)+feat).view(1, -1)
				backpointers.append(bptrs_t)

			#Transitions into the STOP Tag for final execution of transition
			terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
			best_tag_id = argmax(terminal_var)
			path_score = terminal_var[0][best_tag_id]

			#Follow the backpointers to determine what is the best path to take
			best_path = [best_tag_id]
			for bptrs_t in reversed(backpointers):
				best_tag_id = bptrs_t[best_tag_id]
				best_path.append(best_tag_id)

			#Pop off the start tag, we don't want to return that to the caller
			start = best_path.pop()
			assert start == self.tag_to_ix[START_TAG]
			best_path.reverse()
			return path_score, best_path

		def neg_log_likelihood(self, characters, tags):
			feats = self._get_lstm_features(characters)
			forward_score = self._forward_alg(feats)
			gold_score = self._score_sentence(feats,tags)
			return forward_score - gold_score

		def _score_sentence(self, feats, tags):
			#Give the score of the tagged seq
			score = torch.zeros(1)
			tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])
			for i, feat in enumerate(feats):
				score = score + self.transitions[tags[i+1], tags[i]] + feat[tags[i+1]]
			score = score + self.transitions[self.tag_to_ix[STOP_TAG], tags[-1]]
			return score

		def _forward_alg(self, feats):
			#Forward for partition algorithm
			init_alphas = torch.full((1, self.tagset_size), -10000.)
			#START Tag has all of the score
			init_alphas[0][self.tag_to_ix[START_TAG]] = 0

			#Wrap in variable to get automatic backprop
			forward_var = init_alphas

			#Iterate through the sequence of characters
			for feat in feats:
				alphas_t = [] #forward tesnor
				for next_tag in range(self.tagset_size):
					# broadcast the emission score: it is the same regardless of the previous tag
					emit_score = feat[next_tag].view(1, -1).expand(1, self.tagset_size)
					# the ith entry of trans_score is the score of transitioning to next_tag from i
					trans_score = self.transitions[next_tag].view(1, -1)
					# The ith entry of next_tag_var is the value for the edge (i -> next_tag) before we do log-sum-exp
					next_tag_var = forward_var + trans_score + emit_score
					# The forward variable for this tag is log-sum-exp of all the scores.
					alphas_t.append(log_sum_exp(next_tag_var).view(1))
				forward_var = torch.cat(alphas_t).view(1,-1)
			terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
			alpha = log_sum_exp(terminal_var)
			return alpha


if __name__ == "__main__":
	#Initialize these Variables
	START_TAG = "<"
	STOP_TAG = ">"
	EMBEDDING_DIM = 4
	HIDDEN_DIM = 4

	#Example Data
	training_data = [(list("กฎหมายกับการเบียดบังคนจน"), list("000001010010000010101"))]

	char_to_ix = {}
	for sentence, tags in training_data:
		for character in sentence:
			if character not in char_to_ix:
				char_to_ix[character] = len(char_to_ix)

	tag_to_ix = {"0": 0, "1":1, START_TAG: 2, STOP_TAG: 3} #START_TAG: 3, STOP_TAG: 4
	model = BiLSTM_CRF(len(char_to_ix), tag_to_ix, EMBEDDING_DIM, HIDDEN_DIM)
	optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)

	#Check Predictions Before Training
	with torch.no_grad():
    	precheck_sent = prepare_sequence(training_data[0][0], word_to_ix)
    	precheck_tags = torch.tensor([tag_to_ix[t] for t in training_data[0][1]], dtype=torch.long)
    	print(model(precheck_sent))

    #Prepare Sequence from LSTM 

	






