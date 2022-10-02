import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class ChinaAStock():
    def __init__(self, config=None):
        self._config = config
        self.current_index = 0
        self.max_index = self.prepare_data()

    def prepare_data(self):
        """将price_TI_data数据加载到内存中"""
        if not self._config['std_data']:
            X_path = "price_TI_data/original/"
        else:
            X_path = "price_TI_data/normalized/"
        self.X_train = np.load(X_path + "X_train.npy")
        self.X_val = np.load(X_path + "X_val.npy")
        self.X_test = np.load(X_path + "X_test.npy")
        Y_path = "price_TI_data/Y/"
        self.Y_train = np.load(Y_path + "Y_train.npy")
        self.Y_val = np.load(Y_path + "Y_val.npy")
        self.Y_test = np.load(Y_path + "Y_test.npy")
        self.train_size = self.X_train.shape[0]
        if self._config['choose_feat'] != 'all':
            self.X_train = self.X_train[:, :, :, self._config['choose_feat']]
            self.X_val = self.X_val[:, :, :, self._config['choose_feat']]
            self.X_test = self.X_test[:, :, :, self._config['choose_feat']]
        if not self._config['y_by_window']:
            self.Y_train = self.Y_train[:, 0, :]
            self.Y_val = self.Y_val[:, 0, :]
            self.Y_test = self.Y_test[:, 0, :]
        if not self._config['validation']:
            self.X_test = np.concatenate([self.X_val, self.X_test])
            self.Y_test = np.concatenate([self.Y_val, self.Y_test])
            self.X_val = None
            self.Y_val = None
        if not self._config['validation']:
            data_len = self.Y_train.shape[0] + self.Y_test.shape[0]
        else:
            data_len = self.Y_train.shape[0] + self.Y_val.shape[0] + self.Y_test.shape[0]
        return data_len

    def prepare_news(self, news_data):
        """将news数据加载到内存中且与price_TI_data合并为X"""
        self.news_train = np.load(news_data + '/news_train.npy')
        self.news_val = np.load(news_data + '/news_val.npy')
        self.news_test = np.load(news_data + '/news_test.npy')

        def normalize_4orderTensor(X):
            """对embedding的每个维度各自做normalize"""
            for i in range(X.shape[0]):
                for j in range(X.shape[1]):
                    # 直接对列进行norm
                    for k in range(X.shape[3]):
                        norm = np.linalg.norm(X[i, j, :, k])
                        if norm != 0:
                            X[i, j, :, k] = X[i, j, :, k] / norm
            return X

        if self._config['std_data']:
            self.news_train = normalize_4orderTensor(self.news_train)
            self.news_val = normalize_4orderTensor(self.news_val)
            self.news_test = normalize_4orderTensor(self.news_test)
        if not self._config['validation']:
            self.news_test = np.concatenate([self.news_val, self.news_test])
            self.news_val = None

            self.X_train = np.concatenate([self.X_train, self.news_train], axis=3)
            self.X_test = np.concatenate([self.X_test, self.news_test], axis=3)
        else:
            self.X_train = np.concatenate([self.X_train, self.news_train], axis=3)
            self.X_val = np.concatenate([self.X_val, self.news_val], axis=3)
            self.X_test = np.concatenate([self.X_test, self.news_test], axis=3)

    def get_all_X(self):
        if not self._config['validation']:
            return np.concatenate([self.X_train, self.X_test])
        else:
            return np.concatenate([self.X_train, self.X_val, self.X_test])

    def fetch_batch(self):
        """随机取出一个batch的X和Y数据"""
        idxs = []
        for _ in range(self._config['batch_size']):
            idx = self.train_size + 1
            while idx > self.train_size:
                idx = np.random.geometric(self._config['bias'])
            idx = self.train_size - idx
            idxs.append(idx)
        X = self.X_train[idxs]
        y = self.Y_train[idxs]
        return X, y

    def fetch_consecutive_batch(self):
        """随机取出一个连续的batch_size的X和Y数据"""
        idxs = []
        while len(idxs) < self._config['batch_size']:
            idx = self.train_size + 1
            while idx > self.train_size:
                idx = np.random.geometric(self._config['bias'])
            idx = self.train_size - idx

            if idx + self._config['batch_size'] <= self.train_size:
                idxs += list(range(idx, idx + self._config['batch_size']))
            else:
                # 如果在最后几天且天数少于batch_siz
                idxs += list(range(idx, self.train_size))
        X = self.X_train[idxs]
        y = self.Y_train[idxs]
        return X, y

    def backtest(self, pred, on='all', plot=True, save_dir=None):
        y_true = None
        if on == 'all':
            if not self._config['y_by_window']:
                y_true = np.concatenate([self.Y_train, self.Y_val, self.Y_test])
            else:
                y_true = np.concatenate([self.Y_train[:, 0, :], self.Y_val[:, 0, :], self.Y_test[:, 0, :]])
        elif on == 'train':
            if not self._config['y_by_window']:
                y_true = self.Y_train
            else:
                y_true = self.Y_train[:, 0, :]
        elif on == 'val':
            if not self._config['y_by_window']:
                y_true = self.Y_val
            else:
                y_true = self.Y_val[:, 0, :]
        elif on == 'test':
            if not self._config['y_by_window']:
                y_true = self.Y_test
            else:
                y_true = self.Y_test[:, 0, :]

        # 计算收益率和sharpe
        profit = np.sum(pred * y_true, axis=1)
        cum_profit = np.cumsum(profit)  # 所有时间的累计对数收益率
        sharpe_ratio = np.mean(profit) * np.sqrt(252) / np.std(profit)

        print(f'Cumulated {on} logReturn: {cum_profit[-1]}, {on} Sharpe: {sharpe_ratio}')
        # 画回测图
        if plot:
            plt.figure(figsize=(12, 5))
            plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.95)
            plt.plot(cum_profit, label=self._config['config_name'])
            plt.title(f'logReturn:{cum_profit[-1]}')
            plt.legend()
            plt.xlabel('Days')
            plt.ylabel('Cumulative Log Return')
            if save_dir is None:
                plt.savefig(f'{self._config["config_name"]}_{on}.png')
            else:
                plt.savefig(f'{save_dir}_{on}.png')

        if on == 'test':
            # 保存test上的所有时间的日对数收益率
            np.save(f'{save_dir}_dailyProfit.npy', profit)
        return {'cum_profit': cum_profit[-1], 'sharpe': sharpe_ratio}
