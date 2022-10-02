"""根据训练好的模型生成各新闻的embedding"""
import os
import pandas as pd
import numpy as np
import re

import torch
import torch.nn as nn


import model as m


############ Parameters
train_set_percent = 0.8

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


def get_embedding_for_one_news(mdna_section):
	if mdna_section=='':
		#对于empty情形下，如何处理新闻空缺处的embedding
		# embedding=np.ones(lstm_hidden_dim*2)*1e-6#用很小的数
		embedding=np.zeros(lstm_hidden_dim*2)#置0

	else:
		#中文分句
		sent_tokenied_mdna = re.split('[。！？]',mdna_section)
		embedding= hibert_model.encode(sent_tokenied_mdna).detach().cpu().numpy()
	return embedding



## MAIN

news=pd.read_csv('../news_na_empty.csv').iloc[:,:].fillna('')#对empty情形要fillna()
#平铺展开为1维
embedding_all=[]
for day in range(661):
	embeddings_all_stocks_one_day = []
	for stock in range(180):
		embeddings_all_stocks_one_day.append(get_embedding_for_one_news(news.iloc[stock,day]))
	embeddings_all_stocks_one_day=np.array(embeddings_all_stocks_one_day)
	embedding_all.append(embeddings_all_stocks_one_day)
embedding_all=np.array(embedding_all)
np.save('news_embedding_empty_zero.npy',embedding_all)







