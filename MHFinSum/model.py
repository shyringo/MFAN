import torch
import torch.nn as nn

from sentence_transformers import SentenceTransformer
import os



class HiBERT_model(nn.Module):
	def __init__(self, device, sent_embed_size, nHidden,   buy_class = 2, \
				sentence_encoder_model ='distilbert-base-nli-mean-tokens'):
		super(HiBERT_model, self).__init__()
		self.device = device
		print('sentence_encoder_model cache exists: ',os.path.exists(sentence_encoder_model))
		self.sentence_encoder = SentenceTransformer(sentence_encoder_model)

		self.sent_embed_size = sent_embed_size
		self.nHidden = nHidden

		self.lstm = nn.LSTM(sent_embed_size, nHidden, bidirectional = True)

		self.uw =  nn.Parameter(torch.randn(2*nHidden, 1), requires_grad = True)
		self.hidden2context = nn.Linear(2*nHidden, 2*nHidden)

		# Final layer for Buy/Sell prediction
		self.buy_predictor_layer = nn.Linear(2*nHidden, buy_class)

		# Final layer for sector prediction



	def forward(self, sents):
		encoded_sents = torch.tensor(self.sentence_encoder.encode(sents))
		encoded_sents = encoded_sents.view(-1, 1, self.sent_embed_size).to(self.device)

		recurrent, (hidden, c) = self.lstm(encoded_sents)
		recurrent = recurrent.view(-1, 2*self.nHidden)
		ut = torch.tanh(self.hidden2context(recurrent))
		alphas = torch.softmax(torch.mm(ut, self.uw), 0)

		context = torch.sum(recurrent * alphas.expand_as(recurrent), dim=0)

		# Append index return feature to doc representation
		# if self.use_index_movement:
		# 	final_context = torch.cat((context, torch.tensor([index_movement]).to(self.device)))
		# else:
		final_context = context

		# Getting buy classification
		buy_classification = self.buy_predictor_layer(final_context).view(1,-1).to(self.device)

		# Getting sector classification
		# sic_classification = None

		return buy_classification, alphas

		# pdb.set_trace()

	def encode(self,sents):
		encoded_sents = torch.tensor(self.sentence_encoder.encode(sents))
		encoded_sents = encoded_sents.view(-1, 1, self.sent_embed_size).to(self.device)

		recurrent, (hidden, c) = self.lstm(encoded_sents)
		recurrent = recurrent.view(-1, 2*self.nHidden)
		ut = torch.tanh(self.hidden2context(recurrent))
		alphas = torch.softmax(torch.mm(ut, self.uw), 0)

		context = torch.sum(recurrent * alphas.expand_as(recurrent), dim=0)

		return context
	def get_probs_for_labels(self,sents):
		encoded_sents = torch.tensor(self.sentence_encoder.encode(sents))
		encoded_sents = encoded_sents.view(-1, 1, self.sent_embed_size).to(self.device)

		recurrent, (hidden, c) = self.lstm(encoded_sents)
		recurrent = recurrent.view(-1, 2*self.nHidden)
		ut = torch.tanh(self.hidden2context(recurrent))
		alphas = torch.softmax(torch.mm(ut, self.uw), 0)

		context = torch.sum(recurrent * alphas.expand_as(recurrent), dim=0)

		final_context = context
		buy_classification = self.buy_predictor_layer(final_context).view(1,-1)
		probs=torch.softmax(buy_classification[0],0).to(self.device)
		return probs


