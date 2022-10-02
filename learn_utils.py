import tensorflow as tf

def ReturnLoss(y_true,y_pred):
    loss = -tf.reduce_mean(tf.reduce_sum(tf.multiply(y_pred,y_true),axis=1))
    return loss

def SharpeLoss(y_true,y_pred):
    rt = tf.reduce_sum(tf.multiply(y_pred,y_true),axis=1)
    var = tf.math.reduce_std(rt)
    loss = -252**0.5 * tf.reduce_mean(rt) / var
    return loss

def RPLoss5(y_true,y_pred):
    rt = tf.reduce_sum(tf.multiply(y_pred,y_true),axis=1)
    var = tf.math.reduce_std(rt)
    loss = -252**0.5 * tf.reduce_mean(rt) + 0.5 * var
    return loss

def RPLoss1(y_true,y_pred):
    rt = tf.reduce_sum(tf.multiply(y_pred,y_true),axis=1)
    var = tf.math.reduce_std(rt)
    loss = -252**0.5 * tf.reduce_mean(rt) + 0.1 * var
    return loss

def RPLoss10(y_true,y_pred):
    rt = tf.reduce_sum(tf.multiply(y_pred,y_true),axis=1)
    var = tf.math.reduce_std(rt)
    loss = -252**0.5 * tf.reduce_mean(rt) + 1 * var
    return loss

def RPLoss20(y_true,y_pred):
    rt = tf.reduce_sum(tf.multiply(y_pred,y_true),axis=1)
    var = tf.math.reduce_std(rt)
    loss = -252**0.5 * tf.reduce_mean(rt) + 2 * var
    return loss

class RiskPreferenceLoss(tf.keras.losses.Loss):
    def __init__(self,rp=10,window=10,**kwarg):
        super(RiskPreferenceLoss, self).__init__()
        self.rp = rp
        self.window = tf.constant(window,dtype=tf.float32)
        self.annualize_factor = tf.constant(252,dtype=tf.float32)

    def call(self, y_true, y_pred):
        y_pred = tf.expand_dims(y_pred,1)
        daily_annual = tf.reduce_sum(tf.multiply(y_pred,y_true),axis=2) * self.annualize_factor
        ar = tf.reduce_mean(daily_annual,axis=1)
        risk_loss = tf.math.reduce_std(daily_annual,axis=1)
        loss = risk_loss*self.rp - ar
        return loss

class MeanVarianceLoss(tf.keras.losses.Loss):
    def __init__(self,AR_target=0.1,window=10,**kwarg):
        super(MeanVarianceLoss, self).__init__()
        self.ar_target = tf.constant(AR_target,dtype=tf.float32)
        self.window = tf.constant(window,dtype=tf.float32)
        self.annualize_factor = tf.constant(252,dtype=tf.float32)

    def call(self, y_true, y_pred):
        y_pred = tf.expand_dims(y_pred,1)
        daily_annual = tf.reduce_sum(tf.multiply(y_pred,y_true),axis=2) * self.annualize_factor
        ar = tf.reduce_mean(daily_annual,axis=1)
        risk_loss = tf.math.reduce_std(daily_annual,axis=1)
        loss = tf.where(ar>self.ar_target,risk_loss,risk_loss+self.ar_target-ar)
        return loss
