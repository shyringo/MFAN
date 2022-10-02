from env import ChinaAStock
from agent import MultiFactorAttention
from controller import AgentController
import tensorflow as tf
from learn_utils import *
import os
import numpy as np
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
tf.config.set_soft_device_placement(True)
physical_devices = tf.config.experimental.list_physical_devices('GPU')
if len(physical_devices) > 0:
    for device in physical_devices:
        tf.config.experimental.set_memory_growth(device, True)

# for now we use a defined dict to replace config file
my_config = {
    'window': 10,
    'n_stocks': 180,
    'y_by_window':False,#是否使用Y的滑动窗口维度
    'learning_rate': 5e-5,
    'batch_size':32,
    'shuffle':False,
    'plot':True,
    'plot_baseline':[],
    'save_pv':True,
    'save_pv_name':'attention_out',
    'bias':0.001,
    'max_steps':100,#控制训练几个batch
    'validation':False,#若为False则用val+test一块只作为test
    'check_interval':200,#val时每隔几步尝试更新best
    'test_loss':None,
    'check_threshold':-99,
    'd_embedding':32
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
    'mv0':MeanVarianceLoss(0),
    'mv02':MeanVarianceLoss(0.1),
    'mv05':MeanVarianceLoss(0.5),
    'mv10':MeanVarianceLoss(1),
    'rt':ReturnLoss,
    'sr' : SharpeLoss,
    'rp05':RPLoss5,
    'rp10':RPLoss10,
    'rp20':RPLoss20,
}
if __name__ == '__main__':
    my_config['model'] = 'MFAN'
    my_config['std_data'] = True#控制数据是否归一化
    for n_feat in [8]:
        my_config['n_feat'] = n_feat
        my_config['choose_feat'] = cfeat[n_feat]
        for loss in ['mv10']:
            my_config['loss'] = loss_dict[loss]
            for learning_rate in [5e-5]:
                my_config['learning_rate']=learning_rate
                for news_data in ['news_data_last_zero']:
                    for trial in range(3):
                        my_config['config_name'] = f'{my_config["model"]}_{news_data}_trial{str(trial)}'
                        print('running',my_config['config_name'] )

                        ac = AgentController(my_config)
                        ac.train_with_news(news_data)