import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_probability as tfp
from tf_mnf.layers.mnf_dense import MNFDense

def myacc(y_true, y_pred):
    delta = tf.math.abs(y_pred - y_true) / (1 + y_true)
    return tf.reduce_mean(tf.cast(delta <= 0.02, tf.float32))

def median(x):
    return tfp.stats.percentile(x, 50.)

def sigma_nmad(y_true, y_pred):
    del_z = y_pred - y_true
    sigma_nmad = 1.48 * median(tf.math.abs((del_z - median(del_z))/(1 + y_true)))
    return sigma_nmad

def negloglik(y, params):
    rv_y = tfp.distributions.Normal(loc=params[:, :1],
                                    scale=1e-4 + tf.math.softplus(params[:, 1:] * 0.01))
    return -rv_y.log_prob(y)

def ResNetBlock_1d(inputs, channels, down_sample=False, kernel_size=7,
                init='he_normal'):

    __strides = [2, 1] if down_sample else [1, 1]

    x = layers.Conv1D(channels, strides=__strides[0],
                      kernel_size=kernel_size, padding='same', kernel_initializer=init)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv1D(channels, strides=__strides[1],
                      kernel_size=kernel_size, padding='same',
                      kernel_initializer=init)(x)
    x = layers.BatchNormalization()(x)

    if down_sample:
        res = layers.Conv1D(channels, strides=2,
                            kernel_size=1, padding='same',
                            kernel_initializer=init)(inputs)
    elif inputs.shape[-1] != channels:
        res = layers.Conv1D(channels, kernel_size=1, strides=1, padding='same')(inputs)
    else:
        res = inputs

    x = layers.Add()([x, res])
    outputs = layers.ReLU()(x)
    return outputs

def ResNet_1d(spec_res):
    inputs = tf.keras.Input(shape=(spec_res, 2))
    
    conv = layers.Conv1D(32, strides=1, kernel_size=7, 
                         padding='same')(inputs)
    conv = layers.MaxPooling1D()(conv)
    
    conv = ResNetBlock_1d(conv, 64) # 32
    conv = ResNetBlock_1d(conv, 64, down_sample=True)
    
    conv = ResNetBlock_1d(conv, 128,) # 16
    conv = ResNetBlock_1d(conv, 128,  down_sample=True)
    
    conv = ResNetBlock_1d(conv, 256) #8
    conv = ResNetBlock_1d(conv, 256,  down_sample=True)
    
    conv = ResNetBlock_1d(conv, 512) # 4
    conv = ResNetBlock_1d(conv, 512,  down_sample=True)
    
    pool = layers.GlobalAveragePooling1D()(conv)
    pool = layers.Dropout(0.2)(pool)
    
    outputs = layers.Dense(1)(pool)
    
    model = tf.keras.Model(inputs, outputs)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss='logcosh', metrics=[myacc, sigma_nmad])
    
    return model

def base_model(model_dir):
    model = tf.keras.models.load_model(model_dir, compile=False)
    
    inputs = model.input
    outputs = model.layers[-3].output
    
    base_model = tf.keras.Model(inputs, outputs)
    
    return base_model

def MNF_model(base_model, trainable=False):
    inputs = base_model.input
    output_layer = base_model.layers[-1].output
    
    for layer in base_model.layers:
        layer.trainable = trainable
        layer._name = layer.name + '_base'
        
    dense = MNFDense(256)(output_layer)
    dense = tf.keras.layers.BatchNormalization()(dense)
    dense = tf.keras.layers.LeakyReLU()(dense)

    dense = MNFDense(128)(dense)
    dense = tf.keras.layers.BatchNormalization()(dense)
    dense = tf.keras.layers.LeakyReLU()(dense)

    params = tf.keras.layers.Dense(2)(dense)

    model = tf.keras.Model(inputs, params)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-4),
                  loss=negloglik, metrics=[myacc, sigma_nmad])
    return model