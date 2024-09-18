import numpy as np
import os
import tensorflow as tf
import copy 
import re
import pandas as pd
from scipy.interpolate import interp1d

def spectra_augmentation(spectra, aug_n=50):
    
    spec_aug = []
    for _ in range(aug_n - 1):
        spec_temp = copy.deepcopy(spectra)
        spec_temp[:,:,0] = spec_temp[:,:,0] + \
            np.random.normal(0, scale=spec_temp[:,:,1])
        spec_aug.append(spec_temp)
        
    if aug_n == 1:
        data_aug = spectra
        
    else:
        spec_aug = np.array(spec_aug)
        spec_aug = spec_aug.reshape((aug_n - 1) * spectra.shape[0],
                                    spectra.shape[1], spectra.shape[2])
        data_aug = np.row_stack((spectra, spec_aug))
        
    return data_aug

def data_loading(batch_size=1024, augmentation=True, aug_n=50):
    
    ds = np.load('../sls_desi_and_eboss.npy')
    params = pd.read_csv('../sls_desi_and_eboss_with_snr.csv')
    dist = np.loadtxt('../1d_points.d')
    
    interp = interp1d(dist[:, 0], dist[:, 1], kind='linear', 
                      bounds_error=False, fill_value='extrapolate')
    
    redshifts = params['z'].values
    sample_probability = interp(redshifts)
    sample_probability[sample_probability < 0] = 0
    sample_probability = sample_probability/np.sum(sample_probability)
    
    size = redshifts.shape[0]
    
    spectra = ds[:, :, 1:]
    
    spec_res = spectra.shape[1]
    
    all_idx = np.arange(redshifts.shape[0])
    
    seed = 123
    np.random.seed(seed)
    # test_idx = np.random.choice(all_idx, int(all_idx.shape[0] * 0.1), replace=False)
    test_idx = np.random.choice(all_idx, int(all_idx.shape[0] * 0.1), replace=False,
                                p=sample_probability)
    np.save(f'test_idx_seed_{seed}_csst_probability.npy', test_idx)
    
    train_idx = np.setdiff1d(all_idx, test_idx)
    
    val_size = int(redshifts.shape[0] * 0.1)
    
    val_idx = train_idx[:val_size]
    train_idx = train_idx[val_size:]
    
    train_spectra = spectra[train_idx]
    train_z = redshifts[train_idx]
    
    val_spectra = spectra[val_idx]
    val_z = redshifts[val_idx]
    
    shuffle_idx = np.random.choice(train_z.shape[0],
                                       train_z.shape[0], 
                                       replace=False)
    
    if augmentation:
        train_spectra = spectra_augmentation(train_spectra, aug_n=aug_n)
        train_z = np.tile(train_z, aug_n)
        
        shuffle_idx = np.random.choice(train_z.shape[0],
                                       train_z.shape[0], 
                                       replace=False)
        
    train_spectra = train_spectra[shuffle_idx]
    train_z = train_z[shuffle_idx]
    
    with tf.device('CPU'):
        
        train_sls_ds = tf.data.Dataset.from_tensor_slices(train_spectra)
        train_z_ds = tf.data.Dataset.from_tensor_slices(train_z)
        
        train_ds = tf.data.Dataset.zip((train_sls_ds, train_z_ds)).batch(batch_size)
        
        
    test_spectra = spectra[test_idx]
    test_z = redshifts[test_idx]
    
    val_data = val_spectra
    test_data = test_spectra
    
    return train_ds, test_data, test_z, spec_res, val_data, val_z