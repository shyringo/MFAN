"""训练的下游任务为股价涨跌预测，新闻填充方法为empty"""
import os
import pandas as pd 
import numpy as np
from collections import Counter
import re

import torch
import torch.nn as nn

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

import model as m


############ Parameters
lr = 0.0005

batch_size = 64
epochs = 5

trained_models_path = './trained_models_empty/'

sentence_encoder_model =  '/home/suhy/.cache/torch/sentence_transformers/sentence-transformers_distilbert-base-nli-mean-tokens'
sentence_embed_size = 768
lstm_hidden_dim = 16
buy_classes = 2

use_index_movement = False
use_sic_mtl = False
mtl_industry_weight_lambda = 0.1

np.random.seed(0)

############ Model related
# Using gpu if available else cpu
device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
# devices =[torch.device(f'cuda:{i}') for i in [0,3]]
print('device:',device)

hibert_model = m.HiBERT_model(device, sentence_embed_size, lstm_hidden_dim,	buy_classes, sentence_encoder_model)
hibert_model = hibert_model.to(device)
# hibert_model = nn.DataParallel(hibert_model, device_ids=devices).to(device)

# for name, param in hibert_model.named_parameters():
#     if param.requires_grad:
#         print(name, param.data)


# Defining loss function
cross_entropy_criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(hibert_model.parameters(), lr = lr)


def single_epoch_hibert(X,Y, train_flag = True):
	golden_labels = []
	predicted_labels = []
	# golden_labels_sic = []
	# predicted_labels_sic = []
	total_loss = 0
	batch_loss = 0
	count = -1

	# Make gradients 0 before starting
	optimizer.zero_grad()

	# for (mdna_section, buy_label) in data:
	print('ready to compute on data')
	for i in range(len(Y)):
		mdna_section = X[i]
		buy_label = Y[i]
		count = count + 1

		# Preparing labels
		# buy label
		buy_label =  torch.tensor([int(buy_label)]).to(device)
		# sector label

		#中文分句
		sent_tokenied_mdna = re.split('[。！？]',mdna_section)
		buy_classification,sent_alphas = hibert_model(sent_tokenied_mdna)

		buy_loss = cross_entropy_criterion(buy_classification, buy_label)

		final_loss = buy_loss

		# Backpropagation (gradients are accumulated in parameters)
		final_loss.backward()

		# Accumulating the total loss
		loss_data = float(final_loss.data.cpu().numpy())
		# print(loss_data)
		total_loss += loss_data

		if train_flag and ((count + 1) % batch_size == 0):
			# Gradient Descent after all the batch gradients are accumulated
			optimizer.step()
			optimizer.zero_grad()

		# pdb.set_trace()
		# For getting accuracy
		golden_labels.append(buy_label.data.cpu().numpy()[0])
		predicted_labels.append(torch.argmax(buy_classification, dim=1).data.cpu().numpy()[0])
		# golden_labels_sic.append(sic2id[sic])
		# predicted_labels_sic.append(torch.argmax(sic_classification, dim=1).data.cpu().numpy()[0])

	# Final update for remaining datapoints not included in any batch
	if train_flag:
		optimizer.step()
		optimizer.zero_grad()

	avg_loss = total_loss/len(Y)

	# print(len(golden_labels))
	print(Counter(golden_labels))
	# print(len(predicted_labels))
	print(Counter(predicted_labels))
	return avg_loss, golden_labels, predicted_labels



## MAIN

news=pd.read_csv('../news_na_empty.csv').iloc[:,:541]
labels=np.load('../price_TI_data/original/Y_train.npy')[:,0,:]#平铺展开为1维
#平铺展开为1维
# X=news.fillna('[UNK]').to_numpy().T.reshape(-1)#无新闻处用[UNK]代替
X=news.fillna('').to_numpy().T.reshape(-1)
Y=((labels.reshape(-1)>0)*1)
not_empty_indices=np.where(X!='')[0]
X=X[not_empty_indices]
Y=Y[not_empty_indices]

for e in range(epochs):
	print('epoch starting')

	# for each epoch do train, val and test
	avg_train_loss, golden_labels, predicted_labels = single_epoch_hibert(X,Y, train_flag = True)
	print('Avg. train loss per Epoch: ' + str(avg_train_loss) + ' For Epoch: ' + str(e))
	print('Train acc : ' + str(accuracy_score(golden_labels, predicted_labels)) + '|| Train F1: ' + str(f1_score(golden_labels, predicted_labels)))


	loss_folder = 'trainLoss_' + str(avg_train_loss) + '/'

	full_path = trained_models_path + loss_folder
	if not os.path.exists(full_path):
		os.makedirs(full_path)

	torch.save(hibert_model.state_dict(), full_path + 'hibert_model.pkl')

	# _, golden_labels, predicted_labels = single_epoch_hibert(test_data, train_flag = False)
	# report = open(trained_models_path + 'lr' + str(lr) + '_lstm_hidden_' + str(lstm_hidden_dim) + '_index_' + str(use_index_movement) +  '_mtl_' + str(use_sic_mtl) + '_lambda_' + str(mtl_industry_weight_lambda) + '_report.txt', 'a')
	# report.write('Epoch: ' + str(e) + '\n')
	# report.write('Avg. train loss per Epoch: ' + str(avg_train_loss) + ' For Epoch: ' + str(e) + '\n')
	# report.write('Avg. validation loss per Epoch: ' + str(avg_val_loss) + ' For Epoch: ' + str(e) + '\n')
	# report.write('MCC: ' + str(matthews_corrcoef(golden_labels, predicted_labels)) + '\n')
	# report.write(classification_report(golden_labels, predicted_labels))
	# report.write('='*10 + '\n')
	# report.close()

