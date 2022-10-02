import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt


# stocks180=pickle.load(open('price_TI_data/stock_180.pkl','rb'))
# news_dates=pickle.load(open('news_dates.pkl','rb'))
# df_news_stocks=pd.read_csv('news_stocks.csv',dtype=str)

# #缺的新闻用空填充
# for stock in stocks180:
#     df_this_stock=df_news_stocks[df_news_stocks['Symbol']==stock]
#     for i in df_this_stock.index:
#         if df_this_stock.loc[i,'DeclareDate'] in news_dates:
#             if len(df_news.loc[stock,df_this_stock.loc[i,'DeclareDate']])==0:
#                 df_news.loc[stock,df_this_stock.loc[i,'DeclareDate']]=df_this_stock.loc[i,'NewsContent']
#             else:
#                 df_news.loc[stock, df_this_stock.loc[i, 'DeclareDate']]=df_news.loc[stock,df_this_stock.loc[i,'DeclareDate']]+' '+df_this_stock.loc[i,'NewsContent']
#     print()
# df_news.to_csv('news_na_empty.csv',index=False)

#缺的新闻去找最近的last一条填充
# df_news_empty=pd.read_csv('news_na_empty.csv').fillna('')
# df_news_empty.index=stocks180
# for stock in df_news_empty.index:
#     for date in df_news_empty.columns:
#         if df_news_empty.loc[stock,date]=='':
#             df_this_stock_news=df_news_stocks[df_news_stocks['Symbol']==stock]
#             which=sorted(df_this_stock_news['DeclareDate'].to_list()+[date]).index(date)
#             if which!=0:
#                 df_news_empty.loc[stock, date] =df_this_stock_news.iloc[which-1,3]
# df_news_empty.to_csv('news_na_last.csv',index=False)


# # 给news增加时间窗口维度，成为4阶张量,并拆分为train,val,test
# news_embedding_paths=['news_embedding64_empty_zero.npy','news_embedding128_empty_zero.npy']
# for news_embedding_path in news_embedding_paths:
#     news=np.load('MHFinSum/'+news_embedding_path)
#     d_embedding=news.shape[-1]
#     all=[]
#     for i in range(661):
#         ten_days=[]
#         for j in range(-9,1):
#             target=i+j
#             if target>=0:
#                 ten_days.append(news[target,:,:])
#             else:
#                 ten_days.append(np.zeros((180,d_embedding)))#最开始几天都置0
#                 # ten_days.append(np.ones((180, 32))*1e-6)  # 最开始几天都置1e-6
#                 # ten_days.append(news[0,:,:])#用第1天的新闻向前填充
#         all.append(ten_days)
#     all=np.array(all)
#     np.save(news_embedding_path[:-4]+'/news_train.npy',all[:541,:,:,:])
#     np.save(news_embedding_path[:-4]+'/news_val.npy',all[541:601,:,:,:])
#     np.save(news_embedding_path[:-4]+'/news_test.npy',all[601:,:,:,:])


# a=np.load('price_TI_data/Y/Y_train.npy')
# b=np.load('price_TI_data/Y/Y_val.npy')
# c=np.load('price_TI_data/Y/Y_test.npy')
# a=np.load('MFAN_embedding_64_trial2/saved_preds.npy')


# #news最开始几天无新闻的2个stock去csmar找原始新闻数据填充
# df_news_last=pd.read_csv('news_na_last.csv').fillna('')
# text68=open('300039.txt','r',encoding='utf-8').read()
# text71=open('300127.txt','r',encoding='utf-8').read()
# for day in df_news_last.columns:
#     if df_news_last.loc[68,day]=='':
#         df_news_last.loc[68,day]=text68
# for day in df_news_last.columns:
#     if df_news_last.loc[71,day]=='':
#         df_news_last.loc[71,day]=text71
# df_news_last.to_csv('news_na_last.csv',index=False)

# #将news_data四阶张量normalize
# X_train=np.load('news_data_last_day1/news_train.npy')
# for i in range(X_train.shape[0]):
#     for j in  range(X_train.shape[1]):
#         #直接对列进行norm
#         for k in range(X_train.shape[3]):
#             norm = np.linalg.norm(X_train[i,j,:,k])
#             X_train[i,j,:,k] = X_train[i,j,:,k] / norm
#         # #直接对矩阵进行norm
#         # norm = np.linalg.norm(X_train[i,j,:],1)
#         # X_train[i,j,:] = X_train[i,j,:] / norm

# #画上证180收益率全图
# sz180 = pd.read_csv('000010.csv')['logReturn'][:]
# sz180.index = range(len(sz180))
# plt.figure(figsize=(9, 4))
# plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95)
# plt.xlabel('Days')
# plt.ylabel('Cumulative Log Return')
# cum_profit = np.cumsum(sz180)
# plt.plot(cum_profit, label='sz180')
# plt.savefig('sz180.png')

#给trial3都取平均存到trialMean
for model_name in ['MFAN_news_last']:
    profit_all=[]
    for trial in range(3):
        profit=np.load(model_name+'_trial'+str(trial)+'/backtest_dailyProfit.npy')
        profit_all.append(profit)
    profit_all=np.array(profit_all)
    profitMean=np.mean(profit_all,axis=0)
    np.save(model_name+'_trialMean/backtest_dailyProfit.npy',profitMean)
print()