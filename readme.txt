MFAN指Multi-Factor Attention model with News。

# 新闻数据
news_dates.pkl是新闻数据的所有日期，从2011-03-11到2013-12-02，共661天。与Y的[0][0]的所有日期一致。训练集是2011-03-11至2013-06-03共541天，测试集是2013-06-04至2013-12-02共120天。
news_stocks.csv是原始新闻数据，列名为NewsID国泰安生成的ID,DeclareDate发布日期,Symbol股票代码,NewsContent标题加空格加正文。按DeclareDate时间顺序排序，从2011-02-21到2013-11-30。主键为(NewsID,Symbol)。股票代码是在180只中的股票的代码。
news_na_empty.csv是将没有新闻的交易日用''填充的二阶矩阵。180只股票*661天。
news_na_last.csv是将没有新闻的交易日用{上一个有新闻的自然日的新闻}填充的二阶矩阵。180只股票*661天。
news_data/中的news_train.npy是(天数541*时间窗口10天*180只*32维)的新闻embedding，虽然恰好取值本身就是0-1之间的，但是大小差距很大。时间窗口是向前取的10天。
news_embedding.npy是(661天*180只*32维)的新闻embedding

# 价格和TI数据
在文件夹price_TI_data中。
dataset_dates.pkl是X包括train, val, test的[0][0]的所有日期，从2011-02-28到2013-11-19,共661天。
train_dates.pkl是X_train的日期，从2011-02-28到2013-05-21，共541天。
val_dates.pkl是X_val的日期，从2013-05-22到2013-08-16，共60天。
test_dates.pkl是X_test的日期，从2013-08-19到2013-11-19，共60天。
Y的包括train, val, test的[0][0]的所有日期，是从2011-03-14/2011-03-11到2013-12-03/2013-12-02，共661天。Y是以X的当天为第0天时，第10天的收盘价close相对于第9天的对数收益率。
X前4列特征是close, open, high, low。
X_train是(541天*时间窗口10天*股票数180*特征数18)维的四阶张量，Y_train是(541天*时间窗口10天*股票数180)维的三阶张量，但在实际运行中由于y_by_window设置为False用的是Y_train[:,0,:]即去掉了时间窗口维。
stock_180.pkl是这180只对应的股票的代码，按照X[0][0]中的纵向顺序排列，格式为list。

#一些idea
针对SARL，SARL结合了新闻和价格信息，但没有技术指标TI。MFAN就都结合了，新闻则能够捕获投资者的一手信息来源和市场情绪的。将新闻embedding直接concat加入多因子的因子维，灵感来源于SARL直接将新闻embedding concat到state上，但是SARL并没有说policy network有何变化。空缺新闻直接置0的灵感是SARL。
多因子+注意力模型的灵感来源于MFA。
把预测下一日涨跌作为下游任务训练新闻embedding的灵感是MHFinSum。


#MHFinSum
是论文MHFinSum的代码，其中用到的sentence_transformers模型的需要下载的cache,在windows上是在C:\Users\Lenovo\.cache\torch\sentence_transformers\sentence-transformers_distilbert-base-nli-mean-tokens中，在linux上是在/home/suhy/.cache/torch/sentence_transformers/sentence-transformers_distilbert-base-nli-mean-tokens中。