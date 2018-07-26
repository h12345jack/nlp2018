import torch
import torch.nn as nn
import numpy as np 
import pickle
import os

CUR_DIR = os.path.abspath(os.path.dirname(__file__))
START_TAG = "<START>"
STOP_TAG = "<STOP>"


def argmax(vec):
	# return the argmax as a python int
	_, idx = torch.max(vec, 1)
	return idx.item()


# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(vec): # batch x tagset
	# max_score = vec[0, argmax(vec)]
	batch_size = len(vec)
	max_score = vec[range(batch_size), torch.argmax(vec, 1)] # batch
	max_score_broadcast = max_score.view(batch_size, -1).expand(batch_size, vec.size()[1])
	return max_score + \
		torch.log(torch.sum(torch.exp(vec - max_score_broadcast), 1)) # batch


def prepare_sequence(seq, to_idx):
	idxs = [to_idx[w] if w in to_idx else 0 for w in seq]
	return torch.tensor(idxs, dtype=torch.long)


class BiLSTM_CRF(nn.Module):

	def __init__(self, vocab_size, tag_to_ix, embedding_dim, hidden_dim, embeddings):
		super(BiLSTM_CRF, self).__init__()
		self.embedding_dim = embedding_dim
		self.hidden_dim = hidden_dim
		self.vocab_size = vocab_size
		self.tag_to_ix = tag_to_ix
		self.tagset_size = len(tag_to_ix)

		# self.word_embeds = nn.Embedding(vocab_size, embedding_dim)
		self.word_embeds = nn.Embedding.from_pretrained(embeddings, freeze=False)
		self.lstm = nn.LSTM(embedding_dim, hidden_dim // 2,
							num_layers=1, bidirectional=True, batch_first=True)

		# Maps the output of the LSTM into tag space.
		self.hidden2tag = nn.Sequential(
			# nn.Dropout(p=0.2),
			nn.Linear(hidden_dim, hidden_dim),
			nn.ReLU(),
			nn.Linear(hidden_dim, self.tagset_size)
		)
		# self.hidden2tag = nn.Linear(hidden_dim, self.tagset_size)

		# Matrix of transition parameters.  Entry i,j is the score of
		# transitioning *to* i *from* j.
		self.transitions = nn.Parameter(
			torch.randn(self.tagset_size, self.tagset_size))

		# These two statements enforce the constraint that we never transfer
		# to the start tag and we never transfer from the stop tag
		self.transitions.data[tag_to_ix[START_TAG], :] = -10000
		self.transitions.data[:, tag_to_ix[STOP_TAG]] = -10000

		self.hidden = self.init_hidden()

	def init_hidden(self, batch_size=1):
		return (torch.randn(2, batch_size, self.hidden_dim // 2),
				torch.randn(2, batch_size, self.hidden_dim // 2))

	def _forward_alg(self, feats):
		batch_size = len(feats)
		seq_len = len(feats[0])
		# Do the forward algorithm to compute the partition function
		init_alphas = torch.full((batch_size, self.tagset_size), -10000.)
		# START_TAG has all of the score.
		init_alphas[:,self.tag_to_ix[START_TAG]] = 0.

		# Wrap in a variable so that we will get automatic backprop
		forward_var = init_alphas # batch x tagset

		# Iterate through the sentence
		for i in range(seq_len):
			feat = feats[:,i,:]
		# for feat in feats:
			alphas_t = []  # The forward tensors at this timestep
			for next_tag in range(self.tagset_size):
				# broadcast the emission score: it is the same regardless of
				# the previous tag
				# emit_score = feat[next_tag].view(
				#	 1, -1).expand(1, self.tagset_size)
				emit_score = feat[:,next_tag].view(
					batch_size, -1).expand(batch_size, self.tagset_size) # batch x tagset
				# the ith entry of trans_score is the score of transitioning to
				# next_tag from i
				# trans_score = self.transitions[next_tag].view(1, -1)
				trans_score = self.transitions[next_tag].view(
					1, -1).expand(batch_size, self.tagset_size) # batch x tagset
				# The ith entry of next_tag_var is the value for the
				# edge (i -> next_tag) before we do log-sum-exp
				next_tag_var = forward_var + trans_score + emit_score
				# The forward variable for this tag is log-sum-exp of all the
				# scores.
				alphas_t.append(log_sum_exp(next_tag_var).view(batch_size))
			# forward_var = torch.cat(alphas_t).view(1, -1)
			forward_var = torch.stack(alphas_t, 1)
		terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]].view(
			1, -1).expand(batch_size, self.tagset_size)
		alpha = log_sum_exp(terminal_var)
		return alpha # batch

	def _get_lstm_features(self, sentence):
		batch_size = len(sentence)
		seq_len = len(sentence[0])
		self.hidden = self.init_hidden(batch_size)
		embeds = self.word_embeds(sentence) # batch x len x emb
		lstm_out, self.hidden = self.lstm(embeds, self.hidden) # batch x len x hidden
		# lstm_out = lstm_out.view(len(sentence), self.hidden_dim)
		lstm_feats = self.hidden2tag(lstm_out) # batch x len x tagset
		return lstm_feats

	def _score_sentence(self, feats, tags):
		batch_size = len(feats)
		seq_len = len(feats[0])
		# Gives the score of a provided tag sequence
		score = torch.zeros(batch_size)
		# tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]], dtype=torch.long), tags])
		tags = torch.cat([torch.tensor([self.tag_to_ix[START_TAG]]*batch_size, 
			dtype=torch.long).view(batch_size, -1), tags], 1)
		# for i, feat in enumerate(feats):
		for i in range(seq_len):
			# score = score + \
			#	 self.transitions[tags[i + 1], tags[i]] + feat[tags[i + 1]]
			feat = feats[:,i]
			score = score + \
				self.transitions[tags[:,i+1], tags[:,i]] + \
				feat[range(batch_size), tags[:,i+1]]
		score = score + self.transitions[[self.tag_to_ix[STOP_TAG]]*batch_size, tags[:,-1]]
		return score

	def _viterbi_decode(self, feats):
		feats = feats[0]
		backpointers = []

		# Initialize the viterbi variables in log space
		init_vvars = torch.full((1, self.tagset_size), -10000.)
		init_vvars[0][self.tag_to_ix[START_TAG]] = 0

		# forward_var at step i holds the viterbi variables for step i-1
		forward_var = init_vvars
		for feat in feats:
			bptrs_t = []  # holds the backpointers for this step
			viterbivars_t = []  # holds the viterbi variables for this step

			for next_tag in range(self.tagset_size):
				# next_tag_var[i] holds the viterbi variable for tag i at the
				# previous step, plus the score of transitioning
				# from tag i to next_tag.
				# We don't include the emission scores here because the max
				# does not depend on them (we add them in below)
				next_tag_var = forward_var + self.transitions[next_tag]
				best_tag_id = argmax(next_tag_var)
				bptrs_t.append(best_tag_id)
				viterbivars_t.append(next_tag_var[0][best_tag_id].view(1))
			# Now add in the emission scores, and assign forward_var to the set
			# of viterbi variables we just computed
			forward_var = (torch.cat(viterbivars_t) + feat).view(1, -1)
			backpointers.append(bptrs_t)

		# Transition to STOP_TAG
		terminal_var = forward_var + self.transitions[self.tag_to_ix[STOP_TAG]]
		best_tag_id = argmax(terminal_var)
		path_score = terminal_var[0][best_tag_id]

		# Follow the back pointers to decode the best path.
		best_path = [best_tag_id]
		for bptrs_t in reversed(backpointers):
			best_tag_id = bptrs_t[best_tag_id]
			best_path.append(best_tag_id)
		# Pop off the start tag (we dont want to return that to the caller)
		start = best_path.pop()
		assert start == self.tag_to_ix[START_TAG]  # Sanity check
		best_path.reverse()
		return path_score, best_path

	def neg_log_likelihood(self, sentence, tags):
		feats = self._get_lstm_features(sentence) # batch x len x tagset
		forward_score = self._forward_alg(feats)
		gold_score = self._score_sentence(feats, tags)
		return torch.sum(forward_score - gold_score)

	def forward(self, sentence):  # dont confuse this with _forward_alg above.
		# Get the emission scores from the BiLSTM
		lstm_feats = self._get_lstm_features(sentence)

		# Find the best path, given the features.
		score, tag_seq = self._viterbi_decode(lstm_feats)
		return score, tag_seq


class BiLSTMCRF(object):
	def __init__(self, dataset):
		print(CUR_DIR+'/model/'+dataset+'.pth')
		self.model = torch.load(CUR_DIR+'/model/'+dataset+'.pth')
		self.model.eval()
		file = open(CUR_DIR+'/model/'+dataset+'.w2i', 'rb')
		self.word2idx = pickle.load(file)
		file = open(CUR_DIR+'/model/'+dataset+'.i2t', 'rb')
		self.idx2tag = pickle.load(file)

	def cut(self, sentence):
		with torch.no_grad():
			inputs = torch.zeros(1, len(sentence), dtype=torch.long)
			inputs[0] = prepare_sequence(sentence, self.word2idx)
			tag_scores, tag_seq = self.model(inputs)
			cuts = ''
			for i, idx in enumerate(tag_seq):
				# idx = torch.argmax(tag_scores[0][i]).item()
				tag = self.idx2tag[idx]
				if tag == 'B':
					cuts += ' ' + sentence[i]
				elif tag == 'M':
					cuts += sentence[i]
				elif tag == 'E':
					cuts += sentence[i] + ' '
				elif tag == 'S':
					cuts += ' ' + sentence[i] + ' '
			return ' '.join(cuts.split())