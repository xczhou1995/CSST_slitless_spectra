import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import gc
import os
from joblib import Parallel, delayed
from scipy.stats import norm

def sigma(z_pred, z_spec):
    del_z = z_pred - z_spec
    sigma_nmad = 1.48 * \
        np.median(np.abs((del_z - np.median(del_z))/(1 + z_spec)))
    return np.around(sigma_nmad, 5)


def eta(z_pred, z_spec):
    delt_z = np.abs(z_pred - z_spec) / (1 + z_spec)
    et = np.sum((delt_z > 0.02)) / np.shape(z_pred)[0] * 100
    # et = (np.shape(np.where(delt_z > 0.15 * (1 + z_spec))[0])[0] / np.shape(z_pred)[0]) * 100
    return np.around(et, 2)

def plot_figure(z_true, z_pred, z_pred_std, zmin, zmax, base_dir):
    
    plt.figure(figsize=(8, 8))
    plt.errorbar(z_true, z_pred, yerr=z_pred_std, fmt='.', c='red',
                 ecolor='lightblue', elinewidth=0.5)
    a = np.arange(3)
    plt.plot(a, a, c='k', alpha=0.5)
    plt.plot(a, 1.005 * a + 0.005, 'k--', alpha=0.5)
    plt.plot(a, 0.995 * a - 0.005, 'k--', alpha=0.5)
    sigma_all = sigma(z_pred, z_true)
    eta_all = eta(z_pred, z_true)
    plt.xlim(zmin, zmax)
    plt.ylim(zmin, zmax)
    plt.title('$\eta = $' + str(eta_all) +
              '  $\sigma_{NMAD} = $' + str(sigma_all), fontsize=12)
    plt.xlabel('$z_{true}$')
    plt.ylabel('$z_{pred}$')
    
    figname = 'result'
    
    filename = os.path.join(base_dir, figname)
    
    plt.savefig(filename)
    plt.close()
    
def plot_his(his, base_dir):

    plt.figure(figsize=(8, 6))
    plt.plot(his.history['loss'], label='train loss')
    plt.plot(his.history['val_loss'], label='val loss')
    plt.legend()

    figname = 'loss'

    filename = os.path.join(base_dir, figname)
    plt.savefig(filename)
    plt.close()

    plt.figure(figsize=(8, 6))
    plt.plot(his.history['myacc'], label='train acc')
    plt.plot(his.history['val_myacc'], label='val acc')
    plt.legend()

    figname = 'acc'

    filename = os.path.join(base_dir, figname)
    plt.savefig(filename)
    plt.close()
    
    plt.figure(figsize=(8, 6))
    plt.plot(his.history['sigma_nmad'], label='train sigma')
    plt.plot(his.history['val_sigma_nmad'], label='val sigma')
    plt.legend()
    
    figname = 'sigma'
    
    filename = os.path.join(base_dir, figname)
    plt.savefig(filename)
    plt.close()
    
    history_filename = os.path.join(base_dir, 'history.npz')
    np.savez_compressed(history_filename, loss=his.history['loss'], val_loss=his.history['val_loss'],
                        acc=his.history['myacc'], val_acc=his.history['val_myacc'],
                        sigma=his.history['sigma_nmad'], val_sigma=his.history['val_sigma_nmad'])

def find_idx(idx, low, high, alpha, true, means, sigmas):
    ll, lh = norm.interval(low, means[idx], alpha * sigmas[idx])
    hl, hh = norm.interval(high, means[idx], alpha * sigmas[idx])
    if (lh < true[idx] < hh) or (hl < true[idx] < ll):
        return idx


def curve(true, mubar, std, alpha=1, bin_size=0.05):
    CI = np.arange(0.0, 1.0 + bin_size, bin_size)
    indexes = []
    for low, high in zip(CI[0:], CI[1:]):
        # idx = []
        # for i in range(true.shape[0]):
        #     ll, lh = norm.interval(low, means[i], alpha * sigmas[i])
        #     hl, hh = norm.interval(high, means[i], alpha * sigmas[i])
        #     if (lh < true[i] < hh) or (hl < true[i] < ll):
        #         idx.append(i)

        idx = Parallel(n_jobs=20)(delayed(find_idx)(i, low, high, alpha, true, mubar, std)
                                  for i in range(true.shape[0]))
        idx = np.array(idx)
        idx = idx[idx != None].astype('int')
        indexes.append(idx)

    coverage_probability = []
    ini = 0
    coverage_probability.append(ini)
    for i in range(int(1.0/bin_size)):
        ini = (ini + len(indexes[i]))
        coverage_probability.append(ini)

    coverage_probability = [p/true.shape[0] for p in coverage_probability]

    return coverage_probability


def qq_plot(z_true, mubar, std, base_dir):

    bin_size = 0.05
    ci = np.arange(0.0, 1.0 + bin_size, bin_size)

    cov = curve(z_true, mubar, std, alpha=1, bin_size=bin_size)

    plt.figure(figsize=(6, 6))
    plt.plot(ci, cov)
    plt.plot(np.arange(2), linestyle='--', c='k')
    plt.xlabel('Confidence interval')
    plt.ylabel('Coverage probability')

    figname = 'q_q'

    filename = os.path.join(base_dir, figname)
    plt.savefig(filename)
    plt.close()
    
def softplus(x):
    return np.log(np.exp(x) + 1)

def prediction_bnn(test_data, test_label, base_dir, model, saved_weights, n_runs):
    model.load_weights(saved_weights)

    zmax = np.max(test_label)
    zmin = np.min(test_label)

    # test_grz = []
    # test_w1w2 = []
    # for filename in test_filenames:
    #     file = np.load(filename)
    #     test_grz.append(np.transpose(file['grz'], (1, 2, 0)).astype(np.float32))
    #     test_w1w2.append(np.transpose(file['w1w2'], (1, 2, 0)).astype(np.float32))
    # test_grz = np.array(test_grz)
    # test_w1w2 = np.array(test_w1w2)

    # print(test_grz.shape, test_w1w2.shape)

    # dd = os.path.join(base_dir, 'pred_n')
    
    # os.makedirs(dd, exist_ok=True)
    
    z_pred_n_runs = np.zeros((n_runs, test_label.shape[0], 2))
    for i in range(n_runs):
        z_pred_n_runs[i, ...] = np.reshape(model.predict(test_data, batch_size=512, verbose=2),
                                           (test_label.shape[0], 2))
        # saved_filename = os.path.join(dd, f'pred_{i}')
        # np.save(saved_filename, z_pred_n_runs[i])
        tf.keras.backend.clear_session()
        gc.collect()

    z_pred_n_runs[:, :, 1] = 1e-4 + softplus(z_pred_n_runs[:, :, 1] * 0.01)
    # z_pred_n_runs = 1e-4 + np.sqrt(np.exp(z_pred_n_runs[:, :, 1] * 0.001))

    mubar = np.average(z_pred_n_runs[:, :, 0], axis=0)
    aleatoric = np.average(z_pred_n_runs[:, :, 1] ** 2, axis=0)
    epistemic = np.average(z_pred_n_runs[:, :, 0] ** 2 - mubar ** 2, axis=0)

    stdsquare = aleatoric + epistemic

    std = np.sqrt(stdsquare)

    data_name = 'result'

    filename = os.path.join(base_dir, data_name)

    np.savez_compressed(filename, z_pred=np.column_stack(
        (mubar, std)), z_true=test_label, aleatoric=aleatoric,
                        epistemic=epistemic)

    plot_figure(test_label, mubar, std, zmin, zmax,
                base_dir)
    qq_plot(test_label, mubar, std, base_dir)
    
def prediction_cnn(saved_model_name, test_data, test_z, base_dir):
    
    model = tf.keras.models.load_model(saved_model_name, compile=False)
    
    pred_z = model.predict(test_data, batch_size=256).reshape(-1)
    
    zmin = np.min(test_z)
    zmax = np.max(test_z)
    
    data_name = 'result'
    
    filename = os.path.join(base_dir, data_name)
    np.savez_compressed(filename, z_pred=pred_z, z_true=test_z)
    
    plot_figure(test_z, pred_z, None, zmin, zmax, base_dir)