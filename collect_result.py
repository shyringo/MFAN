import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

use_UCRP = False  # 等权策略
use_sz180 = False  # 上证180
use_MHFinsum = False

collect_metrics = True  # 整理成一个dataframe
plot = True  # 各模型的对数收益率曲线画到同一张图上

models = ['MFAN_news_last_trialMean', 'MFAN_trialMean']

Y_val = np.load('price_TI_data/Y/Y_val.npy')[:, 0, :]
Y_test = np.load('price_TI_data/Y/Y_test.npy')[:, 0, :]
Y_test = np.concatenate([Y_val, Y_test])
if use_UCRP:
    UCRP = np.mean(Y_test, axis=1)  # UCRP的所有时间的日对数收益率
if use_sz180:
    sz180 = pd.read_csv('000010.csv')['logReturn'][541:]
    sz180.index = range(len(sz180))
if use_MHFinsum:
    mhfinsum = np.load('MHFinSum/top25dailyProfit.npy')

if plot:
    # 各模型的对数收益率曲线画到同一张图上
    plt.figure(figsize=(12, 5))
    plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95)
    plt.xlabel('Days')
    plt.ylabel('Cumulative Log Return')
    figName = ''
    if use_sz180:
        cum_profit = np.cumsum(sz180)
        plt.plot(cum_profit, label='sz180')
        figName += 'sz180+'
    if use_UCRP:
        cum_profit = np.cumsum(UCRP)
        plt.plot(cum_profit, label='UCRP')
        figName += 'UCRP+'
    if use_MHFinsum:
        cum_profit = np.cumsum(mhfinsum)
        plt.plot(cum_profit, label='MHFinSum')
        figName += 'MHFinSum+'
    for model in models:
        profit = np.load(model + '/backtest_dailyProfit.npy')
        cum_profit = np.cumsum(profit)
        # 去掉后缀_trialMean
        if model[-10:] == '_trialMean':
            model = model[:-10]

        plt.plot(cum_profit, label=model)
        figName += model + '+'

    plt.legend()
    plt.savefig(figName[:-1] + '.png')

if collect_metrics:
    all_results = {}


    def criterion(logProfit):
        cumLogReturn = np.sum(logProfit)
        # profit = np.exp(logProfit) - 1#转成百分比收益率
        profit = logProfit  # 直接用对数收益率计算
        annual_return = 252 * profit.mean()
        annual_std = np.sqrt(252) * profit.std()
        sharpe = annual_return / annual_std
        max_dd = (np.cumsum(profit) - pd.Series(np.cumsum(profit)).expanding().max()).min()
        return {'cumLogR': cumLogReturn, 'AR': annual_return, 'AVol': annual_std, 'sharpe': sharpe, 'MD': max_dd}


    if use_sz180:
        cri = criterion(sz180)
        cri['model'] = 'sz180'
        all_results['sz180'] = cri
    if use_UCRP:
        cri = criterion(UCRP)
        cri['model'] = 'UCRP'
        all_results['UCRP'] = cri
    if use_MHFinsum:
        cri = criterion(mhfinsum)
        cri['model'] = 'MHFinSum'
        all_results['MHFinSum'] = cri
    for model in models:
        profit = np.load(model + '/backtest_dailyProfit.npy')
        cri = criterion(profit)
        # 去掉后缀_trialMean
        if model[-10:] == '_trialMean':
            model = model[:-10]

        cri['model'] = model
        all_results[model] = cri

    all_results = pd.DataFrame(all_results).T
    all_results.to_csv(index=False, path_or_buf='news_fill_3trials.csv')

