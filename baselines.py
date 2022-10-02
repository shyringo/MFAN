"""专门用来复现各baseline"""
from env import ChinaAStock
from agent import MultiFactorAttention
from controller import AgentController
import tensorflow as tf
from learn_utils import *
import os
import numpy as np 
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
tf.config.set_soft_device_placement(True)
physical_devices = tf.config.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

# for now we use a defined dict to replace config file
# TODO: read config file
my_config = {
    'window': 10,
    'n_stocks': 180,
    'n_feat': 8,
    'y_by_window':False,
    'learning_rate': 1e-4,
    'batch_size':32,
    'config_name': 'mfa',
    'shuffle':False,
    'plot':True,
    'plot_baseline':[],
    'save_pv':True,
    'save_pv_name':'attention_out',
    'bias':0.001,
    'max_steps':500,
    'validation':False,
    'check_interval':1600,
    'test_loss':None,
    'check_threshold':-99
}
def test_loss(model,x,y):
    y_pred = model.predict(x)
    dar = np.sum(np.multiply(y_pred,y),axis=1) * np.sqrt(252)
    ar = dar.mean()
    std = dar.std() * 0.1
    return f'mean: {ar}, std: {std}'

#使用的特征
cfeat = {
    3:(0,2,3),
    8:(0,2,3,5,7,11,15,16),
    12:(0,2,3,5,7,8,10,11,12,14,15,16),
    18:'all'
}
loss_dict = {
    # 'rpw01' : RiskPreferenceLoss(0.1),
    # 'rpw05':RiskPreferenceLoss(0.5),
    # 'rpw10':RiskPreferenceLoss(1),
    # 'rpw20':RiskPreferenceLoss(2),
    'mv0':MeanVarianceLoss(0),
    'mv02':MeanVarianceLoss(0.1),
    'mv05':MeanVarianceLoss(0.5),
    'mv10':MeanVarianceLoss(1),
    'rt':ReturnLoss,
    'sr' : SharpeLoss,
    'rp05':RPLoss5,
    'rp10':RPLoss10,
    'rp20':RPLoss20,
    # 'rp01':RPLoss1
}

if __name__ == '__main__':
    for model in ['LSTM-EIIE','CNN-EIIE']:
        my_config['model'] = model
        my_config['std_data'] = False
        for n_feat in [3]:
            my_config['n_feat'] = n_feat
            my_config['choose_feat'] = cfeat[n_feat]
            for n_stocks in [180]:
                my_config['n_stocks'] = n_stocks
                for loss in ['rt']:
                    my_config['loss'] = loss_dict[loss]
                    for trial in range(3):
                        my_config['config_name'] = f'{my_config["model"]}_{n_feat}feat_{loss}_trial{str(trial)}'
                        ac = AgentController(my_config)
                        ac.train()