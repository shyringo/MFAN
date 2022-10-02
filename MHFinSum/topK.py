"""用train好的网络输出预测的涨跌，选涨跌中的topK"""
import os
import pandas as pd
import numpy as np
import re

import torch
import torch.nn as nn


import model as m


############ Parameters
train_set_percent = 0.8
K=25#每天至少选几只最可能涨的股票进行投资
lr = 0.0005

batch_size = 64
# epochs = 3#作者原代码为100

trained_models_path = './trained_models_empty/trainLoss_0.6853104394177909/'

sentence_encoder_model =  r'/home/suhy/.cache/torch/sentence_transformers/sentence-transformers_distilbert-base-nli-mean-tokens'
sentence_embed_size = 768
lstm_hidden_dim = 16
buy_classes = 2

use_index_movement = False
use_sic_mtl = False
mtl_industry_weight_lambda = 0.1

np.random.seed(0)

############ Model related
# Using gpu if available else cpu
device = torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
print('device:',device)
# devices =[torch.device(f'cuda:{i}') for i in [0,3]]

hibert_model = m.HiBERT_model(device, sentence_embed_size, lstm_hidden_dim,	buy_classes, sentence_encoder_model)

hibert_model.load_state_dict(torch.load(trained_models_path + 'hibert_model.pkl'))
hibert_model.eval()

hibert_model = hibert_model.to(device)

# hibert_model = nn.DataParallel(hibert_model, device_ids=devices).to(device)

# for name, param in hibert_model.named_parameters():
#     if param.requires_grad:
#         print(name, param.data)


# Defining loss function
cross_entropy_criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(hibert_model.parameters(), lr = lr)


def get_riseProb_for_one_news(mdna_section):
	if mdna_section=='':
		#没有新闻则认为股价不可能涨
		prob=0

	else:
		#中文分句
		sent_tokenied_mdna = re.split('[。！？]',mdna_section)
		buy_classification = hibert_model.get_probs_for_labels(sent_tokenied_mdna)
		prob=buy_classification.detach().cpu().numpy()[1]
	return prob



## MAIN

news=pd.read_csv('../news_na_empty.csv').iloc[:,:].fillna('')#对empty情形要fillna()

portfolio_all=[]
for day in range(541,661):
	riseProb_all_stocks_one_day = []
	for stock in range(180):
		riseProb_all_stocks_one_day.append(get_riseProb_for_one_news(news.iloc[stock,day]))
	threshold=sorted(riseProb_all_stocks_one_day)[K-1]
	riseProb_all_stocks_one_day=np.array(riseProb_all_stocks_one_day)
	invest_num=np.sum(riseProb_all_stocks_one_day>=threshold)
	portfolio=np.where(riseProb_all_stocks_one_day>=threshold,np.ones(180)*1/invest_num,np.zeros(180))
	portfolio_all.append(portfolio)
portfolio_all=np.array(portfolio_all)
# np.save('top25portfolio.npy',portfolio_all)

Y_val=np.load('../price_TI_data/Y/Y_val.npy')[:,0,:]
Y_test = np.load('../price_TI_data/Y/Y_test.npy')[:,0,:]
Y_test=np.concatenate([Y_val,Y_test])
dailyProfit=np.sum(np.multiply(portfolio_all,Y_test),axis=1)
print('shape:',dailyProfit.shape)
np.save('top25dailyProfit.npy',dailyProfit)







