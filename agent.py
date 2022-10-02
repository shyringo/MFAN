"""根据config['model']匹配至对应的模型"""
import pandas as pd
import numpy as np 
from model import MultiFactorAttention,mssacnr_model,mssacnr,mssacn,SimplifiedMultiFactorAttention,cnn,lstm,NormMultiFactorAttention,MultiFactorAttentionWithNews
from learn_utils import MeanVarianceLoss, ReturnLoss
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import RMSprop,Adam


class TradeAgent():
    def __init__(self,config):
        self._config = config
        if config['model'] == 'CNN-EIIE':
            self.model = cnn(self._config['window'],self._config['n_stocks'],self._config['n_feat'])
        if config['model'] == 'LSTM-EIIE':
            self.model = lstm(self._config['window'],self._config['n_stocks'],self._config['n_feat'])
        if config['model'] == 'mfa_complex':
            self.model = MultiFactorAttention(self._config['window'],self._config['n_stocks'],self._config['n_feat'])
        if config['model'] == 'MFA':
            self.model = SimplifiedMultiFactorAttention(self._config['window'],self._config['n_stocks'],self._config['n_feat'])
        if config['model'] == 'nmfa':
            self.model = NormMultiFactorAttention(self._config['window'],self._config['n_stocks'],self._config['n_feat'])
        if config['model'] == 'mssa':
            self.model = mssacn(self._config['window'],self._config['n_stocks'],self._config['n_feat'])
        if config['model'] == 'mssar':
            self.model = mssacnr(self._config['window'],self._config['n_stocks'],self._config['n_feat'])
        if config['model'] == 'MFAN':
            #n_feat要加上新闻embedding的32
            self.model = MultiFactorAttentionWithNews(self._config['window'],self._config['n_stocks'],self._config['n_feat']+self._config['d_embedding'])
        if config['loss'] is not None:
            loss = config['loss']
        else:
            loss = 'mse'
        self.model.compile(loss=loss,optimizer=Adam(learning_rate=self._config['learning_rate']))
        self.early_stopping = EarlyStopping(monitor='val_loss',patience=10,restore_best_weights=True)
        self._pvm = np.zeros((self._config['pvm_size']))

    def train(self,X_train,Y_train,X_val=None,Y_val=None):
        
        return self.model.fit(X_train,Y_train,batch_size=self._config['batch_size'],verbose=0)#keras的train()函数，返回值的history中记录了loss

    def predict(self,X):
        return np.squeeze(self.model.predict(X))

    def set_pvm(self,pvm,idx):
        self._pvm[idx] = pvm

    def save_model(self):
        self.model.save(self._config['config_name'])
        