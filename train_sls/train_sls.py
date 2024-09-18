import numpy as np
import argparse
import tensorflow as tf
from model import *
from spec_loading import *
from testprocess import *
import os
import shutil

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        
    except RuntimeError as e:
        print(e)
        
strategy = tf.distribute.MirroredStrategy()
print('Number of devices:{}'.format(strategy.num_replicas_in_sync))

parser = argparse.ArgumentParser()

parser.add_argument('--model_type', type=str, default='CNN')
parser.add_argument('--batch_size', type=int, default=512)
parser.add_argument('--epochs', type=int, default=200)
parser.add_argument('--augmentation', action='store_true')
parser.add_argument('--num_augments', type=int, default=50)
parser.add_argument('--n_runs', type=int, default=200)
parser.add_argument('--transfer', action='store_true')

args = parser.parse_args()

model_type = args.model_type
batch_size = args.batch_size
epochs = args.epochs
augmentation = args.augmentation
num_augments = args.num_augments
n_runs = args.n_runs
transfer = args.transfer

print('Batch size: ',  batch_size)
print('Epochs: ', epochs)
print('augmentation: ', augmentation)
print('num_augments: ', num_augments)
print('transfer: ', transfer)
print('n_runs: ', n_runs)

print('begin data loading: ')

train_ds, test_data, test_z, spec_res, val_data, val_z = data_loading(batch_size=batch_size, 
                                                                      augmentation=augmentation,
                                                                      aug_n=num_augments)


if model_type == 'CNN':
    model = ResNet_1d(spec_res)
    base_dir = 'CNN_1d'
    os.makedirs(base_dir, exist_ok=True)

    saved_model_name = os.path.join(base_dir, 'Best_model')

    cbk = tf.keras.callbacks.ModelCheckpoint(saved_model_name, monitor='val_loss',
                                            mode='min', save_best_only=True)

    his = model.fit(train_ds, batch_size=batch_size, epochs=epochs, verbose=2, 
                    validation_data=[val_data, val_z], callbacks=[cbk])
    
    plot_his(his, base_dir)
    prediction_cnn(saved_model_name, test_data, test_z, base_dir)
    
elif model_type == 'BNN':
    base_dir = 'BNN_1d'
    os.makedirs(base_dir, exist_ok=True)
    
    model_dir = os.path.join(base_dir, 'BNN_model')
    saved_model_name = 'model_bnn_mnf.tf'
    saved_model_name = os.path.join(model_dir, saved_model_name)

    cbk = tf.keras.callbacks.ModelCheckpoint(saved_model_name, monitor='val_loss',
                                            mode='min', save_best_only=True, save_weights_only=True)
    
    if transfer:
        base_model_dir = 'CNN_1d/Best_model'
        base = base_model(base_model_dir)
        
        model = MNF_model(base, trainable=False)
        
    else:
        empty = ResNet_1d(spec_res)
        empty.save('Empty_model')
        base_model_dir = 'Empty_model'
        base = base_model(base_model_dir)
        
        model = MNF_model(base, trainable=True)
        
    his = model.fit(train_ds, batch_size=batch_size, epochs=epochs, verbose=2, 
                validation_data=[val_data, val_z], callbacks=[cbk])

    plot_his(his, base_dir)
    prediction_bnn(test_data, test_z, base_dir, model, saved_model_name, n_runs)
    
else:
    print('unrecognized model type!!!')