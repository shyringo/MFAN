import tensorflow as tf
from tensorflow.keras.layers import Dense,LSTM,Input,Dot,Concatenate,Conv2D,Conv1D,BatchNormalization,MaxPool2D,LayerNormalization
from tensorflow.keras import Model
from tensorflow.keras.activations import softmax
from tensorflow.keras.optimizers import RMSprop,Adam
from tensorflow.keras.callbacks import EarlyStopping

def attention(factor):
    inter_e = Dense(2, activation = "tanh")(factor)
    e = Dense(1, activation = "relu")(inter_e)
    alphas = softmax(e, axis=1)
    context = Dot(axes=1)([alphas,factor])
    context = BatchNormalization()(context)
    # context = tf.reduce_mean(context,axis=-1)
    return context

def MultiFactorAttention(window, n_stocks,n_feat):
    X = Input(shape=(window, n_stocks,n_feat))
    feat_list = []
    # attention_weights = []
    for i in range(n_feat):
        # a_list = []
        context_list = []
        # densor1 = Dense(16, activation = "tanh")
        # densor2 = Dense(1, activation = "tanh")
        # doter = Dot(axes=1)
        factor = tf.transpose(X[:,:,:,i],[0,2,1])
        for _ in range(n_stocks):
            # context = attention(attention_input,densor1,densor2,doter)
            # context = attention(attention_input)
            # inter_e = Dense(3, activation = "tanh")(factor)
            e = Dense(1, activation = "tanh")(factor)
            alphas = softmax(e, axis=1)
            context = Dot(axes=1)([alphas,factor])
            context = BatchNormalization()(context)
            context_list.append(context)
            # a_list.append(alphas)
        full_context = Concatenate(axis=1)(context_list)
        # full_attention = Concatenate(axis=1)(a_list)
        # attention_weights.append(full_attention)
        # full_context = tf.expand_dims(full_context,1)
        full_context = tf.expand_dims(full_context,-1)
        feat_list.append(full_context)
    if n_feat == 1:
        full_feat = full_context
    else:
        full_feat = Concatenate(axis=-1)(feat_list)
    output = Conv2D(1,(1,window),activation='tanh')(full_feat)
    # output = tf.squeeze(output)
    output = tf.nn.softmax(output[:,:,0,0]*100,axis=1)
    model = Model(inputs=X,outputs=output)
    return model

def SimplifiedMultiFactorAttention(window, n_stocks,n_feat):
    X = Input(shape=(window, n_stocks,n_feat))
    #将X归一化against最后一天的X，如果输入数据已经是normalized那其实不需要这里再归一化了
    x_norm = X / X[:,-1,None,:,:]
    feat_list = []
    for i in range(n_feat):
        factor = tf.transpose(x_norm[:,:,:,i],[0,2,1])
        e = Dense(n_stocks, activation = "relu",kernel_initializer='glorot_uniform',kernel_regularizer="L2")(factor)
        e = LayerNormalization()(e)
        alphas = softmax(e, axis=1)
        full_context = Dot(axes=1)([alphas,factor])
        full_context = tf.expand_dims(full_context,-1)
        feat_list.append(full_context)
    if n_feat == 1:
        full_feat = full_context
    else:
        full_feat = Concatenate(axis=-1)(feat_list)
    output = Conv2D(3,(1,window),activation='relu')(full_feat)
    output = Conv2D(1,(1,1))(output)
    # output = tf.squeeze(output)
    output = tf.nn.softmax(output[:,:,0,0],axis=1)
    model = Model(inputs=X,outputs=output)
    return model

def NormMultiFactorAttention(window, n_stocks,n_feat):
    X = Input(shape=(window, n_stocks,n_feat))
    x_norm = X / X[:,-1,None,:,:]
    feat_list = []
    for i in range(n_feat):
        factor = tf.transpose(x_norm[:,:,:,i],[0,2,1])
        e = Dense(n_stocks, activation = "relu",kernel_initializer='glorot_uniform',kernel_regularizer="L2")(factor)
        e = LayerNormalization()(e)
        alphas = softmax(e, axis=1)
        full_context = Dot(axes=1)([alphas,factor])
        full_context = tf.expand_dims(full_context,-1)
        feat_list.append(full_context)
    if n_feat == 1:
        full_feat = full_context
    else:
        full_feat = Concatenate(axis=-1)(feat_list)
    output = Conv2D(3,(1,window),activation='relu',kernel_initializer='glorot_uniform',kernel_regularizer="L2")(full_feat)
    output = Conv2D(1,(1,1),kernel_initializer='glorot_uniform',kernel_regularizer="L2")(output)
    # output = tf.squeeze(output)
    output = tf.nn.softmax(output[:,:,0,0],axis=1)
    model = Model(inputs=X,outputs=output)
    return model

def mssacnr_model(window,num_assets,n_feat,ker_size_list=[3,6],filter_num=20):
    x_input = Input(shape=(window,num_assets,n_feat))
    att_feature = n_feat + len(ker_size_list)*filter_num
    inception_list = []
    for kernel_size in ker_size_list:
        tmp_layer = Conv2D(filter_num,(kernel_size,1),(1,1),'same',activation="tanh",kernel_initializer='glorot_uniform',kernel_regularizer="L2")(x_input)
        tmp_layer = Conv2D(filter_num,(window,1),(1,1),'valid',activation="tanh",kernel_initializer='glorot_uniform',kernel_regularizer="L2")(tmp_layer)
        inception_list.append(tmp_layer)
    maxpool = MaxPool2D((window,1))(x_input)
    inception_list.append(maxpool)
    x_inter = Concatenate(name='inception_out')(inception_list)
    q = Conv2D(att_feature,(1,1),(1,1),'valid',activation="linear",kernel_initializer='glorot_uniform',kernel_regularizer="L2")(x_inter)
    k = Conv2D(att_feature,(1,1),(1,1),'valid',activation="linear",kernel_initializer='glorot_uniform',kernel_regularizer="L2")(x_inter)
    v = Conv2D(att_feature,(1,1),(1,1),'valid',activation="linear",kernel_initializer='glorot_uniform',kernel_regularizer="L2")(x_inter)
    q = tf.reshape(q, [-1, num_assets, att_feature])
    k = tf.reshape(k, [-1, num_assets, att_feature])
    v = tf.reshape(v, [-1, num_assets, att_feature])
    mask = tf.matmul(q, k, transpose_b=True) / (att_feature**0.5)
    mask = tf.nn.softmax(mask, axis=2)
    mask = tf.matmul(mask, v)
    # mask = tf.reshape(mask, [-1,att_feature,num_assets])
    x_inter = mask[:,None,:,:] + x_inter
    x_inter = tf.reshape(x_inter, [-1, att_feature, num_assets, 1])
    eiie_dense = Conv2D(3,(att_feature,1),(1,1),'valid',activation="tanh",kernel_initializer='glorot_uniform',kernel_regularizer="L2")(x_inter)
    eiie_shape = eiie_dense.get_shape()
    x_inter = tf.reshape(eiie_dense, [-1, eiie_shape[2], 1, eiie_shape[1]*eiie_shape[3]])
    x_inter = Conv2D(1, [1, 1], padding="valid",kernel_initializer='glorot_uniform',kernel_regularizer="L2",activation='tanh')(x_inter)
    x_inter = tf.reshape(x_inter,[-1,eiie_shape[2]])
    x_inter = Dense(num_assets)(x_inter)
    # output = tf.nn.tanh(x_inter)
    # x_inter = tf.nn.relu(tf.squeeze(x_inter))
    output = tf.nn.softmax(x_inter)
    model = Model(inputs=x_input, outputs=output)
    return model

def cnn(window,num_assets,n_feat):
    x_input = Input(shape=(window,num_assets,n_feat))
    x_norm = x_input / x_input[:,-1,None,:,:]
    # x_norm = x_input
    # att_feature = n_feat + len(ker_size_list)*filter_num
    tmp_layer = Conv2D(2,(3,1),(1,1),'valid',activation="relu",kernel_initializer='glorot_uniform',kernel_regularizer="L2")(x_norm)
    eiie_dense = Conv2D(20,(window-3+1,1),(1,1),'valid',activation="relu",kernel_initializer='glorot_uniform',kernel_regularizer="L2")(tmp_layer)
    x_inter = tf.transpose(eiie_dense,[0,2,1,3])
    x_inter = Conv2D(1, [1, 1], padding="valid",kernel_initializer='glorot_uniform')(x_inter)
    output = tf.nn.softmax(tf.squeeze(x_inter))
    model = Model(inputs=x_input, outputs=output)
    return model

def lstm(window,num_assets,n_feat):
    x_input = Input(shape=(window,num_assets,n_feat))
    x_norm = x_input / x_input[:,-1,None,:,:]
    lstm_list = []
    for i in range(n_feat):
        lstm = LSTM(num_assets,kernel_initializer='glorot_uniform',kernel_regularizer="L2")(x_norm[:,:,:,i])
        lstm = tf.expand_dims(lstm,-1)
        lstm_list.append(lstm)
    x_inter = Concatenate(axis=-1)(lstm_list)
    eiie_dense = Dense(8,activation="relu",kernel_initializer='glorot_uniform',kernel_regularizer="L2")(x_inter)
    x_inter = Dense(1,kernel_initializer='glorot_uniform',kernel_regularizer="L2")(eiie_dense)
    output = tf.nn.softmax(tf.squeeze(x_inter))
    model = Model(inputs=x_input, outputs=output)
    return model

def mssacnr(window,num_assets,n_feat,ker_size_list=[3,6],filter_num=20):
    x_input = Input(shape=(window,num_assets,n_feat))
    att_feature = n_feat + len(ker_size_list)*filter_num
    inception_list = []
    for kernel_size in ker_size_list:
        tmp_layer = Conv2D(filter_num,(kernel_size,1),(1,1),'same',activation="tanh",kernel_initializer='glorot_uniform',kernel_regularizer="L2")(x_input)
        tmp_layer = Conv2D(filter_num,(window,1),(1,1),'valid',activation="tanh",kernel_initializer='glorot_uniform',kernel_regularizer="L2")(tmp_layer)
        inception_list.append(tmp_layer)
    maxpool = MaxPool2D((window,1))(x_input)
    inception_list.append(maxpool)
    x_inter = Concatenate(name='inception_out')(inception_list)
    q = Conv2D(att_feature,(1,1),(1,1),'valid',activation="linear",kernel_initializer='glorot_uniform',kernel_regularizer="L2")(x_inter)
    k = Conv2D(att_feature,(1,1),(1,1),'valid',activation="linear",kernel_initializer='glorot_uniform',kernel_regularizer="L2")(x_inter)
    v = Conv2D(att_feature,(1,1),(1,1),'valid',activation="linear",kernel_initializer='glorot_uniform',kernel_regularizer="L2")(x_inter)
    q = tf.reshape(q, [-1, num_assets, att_feature])
    k = tf.reshape(k, [-1, num_assets, att_feature])
    v = tf.reshape(v, [-1, num_assets, att_feature])
    mask = tf.matmul(q, k, transpose_b=True) / (att_feature**0.5)
    mask = tf.nn.softmax(mask, axis=2)
    mask = tf.matmul(mask, v)
    x_inter = mask[:,None,:,:] + x_inter
    x_inter = tf.transpose(x_inter,[0,3,2,1])
    eiie_dense = Conv2D(3,(att_feature,1),(1,1),'valid',activation="tanh",kernel_initializer='glorot_uniform',kernel_regularizer="L2")(x_inter)
    x_inter = tf.transpose(eiie_dense,[0,2,1,3])
    x_inter = Conv2D(1, [1, 1], padding="valid",kernel_initializer='glorot_uniform',kernel_regularizer="L2")(x_inter)
    x_inter = tf.reshape(x_inter,[-1,num_assets])
    x_inter = Dense(num_assets)(x_inter)
    output = tf.nn.softmax(tf.squeeze(x_inter))
    model = Model(inputs=x_input, outputs=output)
    return model

def mssacn(window,num_assets,n_feat,ker_size_list=[3,6],filter_num=20):
    x_input = Input(shape=(window,num_assets,n_feat))
    x_norm = x_input / x_input[:,-1,None,:,:]
    att_feature = n_feat + len(ker_size_list)*filter_num
    inception_list = []
    for kernel_size in ker_size_list:
        tmp_layer = Conv2D(filter_num,(kernel_size,1),(1,1),'same',activation="relu",kernel_initializer='glorot_uniform',kernel_regularizer="L2")(x_norm)
        tmp_layer = Conv2D(filter_num,(window,1),(1,1),'valid',activation="relu",kernel_initializer='glorot_uniform',kernel_regularizer="L2")(tmp_layer)
        inception_list.append(tmp_layer)
    maxpool = MaxPool2D((window,1))(x_norm)
    inception_list.append(maxpool)
    x_inter = Concatenate(name='inception_out')(inception_list)
    q = Conv2D(att_feature,(1,1),(1,1),'valid',activation="linear",kernel_initializer='glorot_uniform',kernel_regularizer="L2")(x_inter)
    k = Conv2D(att_feature,(1,1),(1,1),'valid',activation="linear",kernel_initializer='glorot_uniform',kernel_regularizer="L2")(x_inter)
    v = Conv2D(att_feature,(1,1),(1,1),'valid',activation="linear",kernel_initializer='glorot_uniform',kernel_regularizer="L2")(x_inter)
    q = tf.reshape(q, [-1, num_assets, att_feature])
    k = tf.reshape(k, [-1, num_assets, att_feature])
    v = tf.reshape(v, [-1, num_assets, att_feature])
    mask = tf.matmul(q, k, transpose_b=True) / (att_feature**0.5)
    mask = tf.nn.softmax(mask, axis=2)
    mask = tf.matmul(mask, v)
    x_inter = mask[:,None,:,:] + x_inter
    x_inter = tf.transpose(x_inter,[0,3,2,1])
    eiie_dense = Conv2D(3,(att_feature,1),(1,1),'valid',activation="relu",kernel_initializer='glorot_uniform',kernel_regularizer="L2")(x_inter)
    x_inter = tf.transpose(eiie_dense,[0,2,1,3])
    x_inter = Conv2D(1, [1, 1], padding="valid",kernel_initializer='glorot_uniform',kernel_regularizer="L2")(x_inter)
    output = tf.nn.softmax(tf.squeeze(x_inter))
    model = Model(inputs=x_input, outputs=output)
    return model

def MultiFactorAttentionWithNews(window, n_stocks,n_feat):
    X = Input(shape=(window, n_stocks,n_feat))
    feat_list = []
    for i in range(n_feat):
        factor = tf.transpose(X[:,:,:,i],[0,2,1])
        e = Dense(n_stocks, activation = "relu",kernel_initializer='glorot_uniform',kernel_regularizer="L2")(factor)
        e = LayerNormalization()(e)
        alphas = softmax(e, axis=1)
        full_context = Dot(axes=1)([alphas,factor])
        full_context = tf.expand_dims(full_context,-1)
        feat_list.append(full_context)
    if n_feat == 1:#其实不可能
        full_feat = full_context
    else:
        full_feat = Concatenate(axis=-1)(feat_list)
    output = Conv2D(3,(1,window),activation='relu')(full_feat)
    output = Conv2D(1,(1,1))(output)
    # output = tf.squeeze(output)
    output = tf.nn.softmax(output[:,:,0,0],axis=1)
    model = Model(inputs=X,outputs=output)
    return model
