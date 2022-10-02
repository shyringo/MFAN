import numpy as np
import pandas as pd
from agent import TradeAgent
from env import ChinaAStock
import os


class AgentController():
    def __init__(self, config):
        self._config = config
        self.env = ChinaAStock(config)
        self._config['pvm_size'] = self.env.max_index
        self.agent = TradeAgent(self._config)
        self.loss_curve = np.zeros(self._config['max_steps'])
        if self._config['validation']:
            self.current_best_on_val = self._config['check_threshold']

    def checkpoint(self):
        """若这次的结果好于best则更新best并保存模型"""
        # 返回本次在val上的收益率和sharpe
        val_result = self.evaluate_part(on='val')
        # print(f"Current val logReturn: {val_result['cum_profit']},Current val sharpe: {val_result['sharpe']}, Best sharpe: {self.current_best_on_val}")

        # 若本次在val上的夏普高于best，则更新best并在全部数据集上backtest
        if val_result['sharpe'] > self.current_best_on_val:
            self.current_best_on_val = val_result['sharpe']
            self.agent.save_model()
            self.evaluate_test_saveAllPred(save_dir=self._config['config_name'] + '/backtest')

    def record(self, i):
        if not os.path.exists(self._config['config_name'] + f'/records/'):
            os.makedirs(self._config['config_name'] + f'/records/')
        self.evaluate_all_saveAllPred(save_dir=self._config['config_name'] + f'/records/backtest_{i}')

    def train(self):
        for i in range(self._config['max_steps']):
            # 取一个batch的X,y数据，然后将之用对应的model的train()训练
            X, y = self.env.fetch_batch()
            history = self.agent.train(X, y)

            if ((i + 1) % self._config['check_interval']) == 0:
                if self._config['test_loss'] is None:
                    print(f"Step {i} Loss: {history.history['loss'][0]}")
                else:
                    print(
                        f"Step {i} Loss: {history.history['loss'][0]}, {self._config['test_loss'](self.agent.model, X, y)}")
                    # self.loss_curve[i] = history.history['loss'][0]
                if self._config['validation']:
                    self.checkpoint()
        if not self._config['validation']:
            self.agent.save_model()
            self.evaluate_test_saveAllPred(save_dir=self._config['config_name'] + '/backtest')

    def train_with_news(self, news_data):
        self.env.prepare_news(news_data)
        for i in range(self._config['max_steps']):
            # 取一个batch的X,y数据，然后将之用对应的model的train()训练
            X, y = self.env.fetch_consecutive_batch()
            history = self.agent.train(X, y)

            if ((i + 1) % self._config['check_interval']) == 0:
                if self._config['test_loss'] is None:
                    print(f"Step {i} Loss: {history.history['loss'][0]}")
                else:
                    print(
                        f"Step {i} Loss: {history.history['loss'][0]}, {self._config['test_loss'](self.agent.model, X, y)}")
                    # self.loss_curve[i] = history.history['loss'][0]
                if self._config['validation']:
                    self.checkpoint()
        if not self._config['validation']:
            self.agent.save_model()
            self.evaluate_test_saveAllPred(save_dir=self._config['config_name'] + '/backtest')

    def evaluate_part(self, on='test'):
        result = None
        if on == 'test':
            out = self.agent.predict(self.env.X_test)
            result = self.env.backtest(out, on=on, plot=False)
        if on == 'val':
            out = self.agent.predict(self.env.X_val)
            result = self.env.backtest(out, on=on, plot=False)
        if on == 'train':
            out = self.agent.predict(self.env.X_train)
            result = self.env.backtest(out, on=on, plot=False)

        return result

    def evaluate_all_saveAllPred(self, save_dir=None):
        full_X = self.env.get_all_X()
        out = self.agent.predict(full_X)
        self.env.backtest(out, on='all', plot=self._config['plot'], save_dir=save_dir)
        if self._config['save_pv']:
            np.save(self._config['config_name'] + '/saved_preds.npy', out)

    def evaluate_test_saveAllPred(self, save_dir=None):
        test_pred = self.agent.predict(self.env.X_test)
        self.env.backtest(test_pred, on='test', plot=self._config['plot'], save_dir=save_dir)

        full_X = self.env.get_all_X()
        out = self.agent.predict(full_X)
        np.save(self._config['config_name'] + '/saved_preds.npy', out)

