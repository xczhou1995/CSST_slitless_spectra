import numpy as np
import matplotlib.pyplot as plt
import argparse
from scipy.stats import norm
from scipy.optimize import bisect
import os

parser = argparse.ArgumentParser()

parser.add_argument('--result_file', type=str)

args = parser.parse_args()

result_file = args.result_file

result_dir = '/'.join(result_file.split('/')[:-1])

result = np.load(result_file)

z_true_total = result['z_true']
z_pred_total = result['z_pred'][:, 0]
z_errs_total = result['z_pred'][:, 1]

bin_size = 0.02
CI = np.arange(0, 1.0 + bin_size, bin_size)

def curve(alpha):
    counts = []
    for low, high in zip(CI[0:-1], CI[1:]):
        ll, lh = norm.interval(low, z_pred_total, z_errs_total * alpha)
        hl, hh = norm.interval(high, z_pred_total, z_errs_total * alpha)
        
        idx_1 = (z_true_total > lh) & (z_true_total < hh)
        idx_2 = (z_true_total > hl) & (z_true_total < ll)
        
        idx = idx_1 | idx_2
        counts.append(np.sum(idx))
        
    cp = []
    ini = 0
    cp.append(ini)
    for i in range(len(counts)):
        ini = (ini + counts[i])
        cp.append(ini)
        
    cp = [p/z_true_total.shape[0] for p in cp]
    return cp

def root_func(x):
    return np.sum(curve(x) - np.arange(0.0, 1.0 + bin_size, bin_size))

def calibration():
    ini = root_func(x=1)
    if np.abs(ini) < 1e-3:
        alpha = 1.0
        return alpha
    
    aa = np.arange(0.1, 3.0, 0.2)
    roots = []
    for a in aa:
        roots.append(root_func(a))
    roots = np.array(roots)
    
    index_inverse = np.where(roots > 0)[0][0]
    low = aa[index_inverse - 1]
    high = aa[index_inverse]
    
    alpha = bisect(root_func, low, high, xtol=1e-4)
    return alpha

def beta_calibration(z_true, z_pred, z_errs):
    
    def curve(alpha):
        counts = []
        for low, high in zip(CI[0:-1], CI[1:]):
            ll, lh = norm.interval(low, z_pred, z_errs * alpha)
            hl, hh = norm.interval(high, z_pred, z_errs * alpha)
            
            idx_1 = (z_true > lh) & (z_true < hh)
            idx_2 = (z_true > hl) & (z_true < ll)
            
            idx = idx_1 | idx_2
            counts.append(np.sum(idx))
            
        cp = []
        ini = 0
        cp.append(ini)
        for i in range(len(counts)):
            ini = (ini + counts[i])
            cp.append(ini)
            
        cp = [p/z_true.shape[0] for p in cp]
        return cp
    
    def root_func(x):
        return np.sum(curve(x) - np.arange(0.0, 1.0 + bin_size, bin_size))
    
    ini = root_func(x=1)
    if np.abs(ini) < 1e-3:
        alpha = 1.0
        return alpha
    
    aa = np.arange(0.1, 3.0, 0.2)
    roots = []
    for a in aa:
        roots.append(root_func(a))
    roots = np.array(roots)
    
    index_inverse = np.where(roots > 0)[0][0]
    low = aa[index_inverse - 1]
    high = aa[index_inverse]
    
    alpha = bisect(root_func, low, high, xtol=1e-4)
    
    return alpha

def curve(alpha, z_true, z_pred, z_errs, CI):
    counts = []
    for low, high in zip(CI[0:-1], CI[1:]):
        ll, lh = norm.interval(low, z_pred, z_errs * alpha)
        hl, hh = norm.interval(high, z_pred, z_errs * alpha)
        
        idx_1 = (z_true > lh) & (z_true < hh)
        idx_2 = (z_true > hl) & (z_true < ll)
        
        idx = idx_1 | idx_2
        counts.append(np.sum(idx))
        
    cp = []
    ini = 0
    cp.append(ini)
    for i in range(len(counts)):
        ini = (ini + counts[i])
        cp.append(ini)
        
    cp = [p/z_true.shape[0] for p in cp]
    return cp

alpha_total = beta_calibration(z_true_total, z_pred_total, z_errs_total)

print(alpha_total)

z_errs_total_cal = z_errs_total * alpha_total

plt.figure(figsize=(6, 6))
plt.plot(CI, curve(1, z_true_total, z_pred_total, z_errs_total, CI), label='Before', linestyle='-.')
plt.plot(CI, curve(alpha_total, z_true_total, z_pred_total, z_errs_total, CI), label='After', linestyle='solid')
plt.plot(CI, CI, linestyle='dashed', c='k')
plt.legend(frameon=False, fontsize=14)
plt.xlabel('Confidence Interval', fontsize=16)
plt.ylabel('Coverage Probability', fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.xlim(0, 1.0)
plt.ylim(0, 1.0)
figname = os.path.join(result_dir, 'Calibration.png')
plt.savefig(figname)

def sigma(z_pred, z_spec):
    del_z = z_pred - z_spec
    sigma_nmad = 1.48 * \
        np.median(np.abs((del_z - np.median(del_z))/(1 + z_spec)))
    return np.around(sigma_nmad, 5)


def eta(z_pred, z_spec):
    delt_z = np.abs(z_pred - z_spec) / (1 + z_spec)
    et = np.sum((delt_z > 0.02)) / np.shape(z_pred)[0] * 100
    # et = (np.shape(np.where(delt_z > 0.15 * (1 + z_spec))[0])[0] / np.shape(z_pred)[0]) * 100
    return np.around(et, 3)

sigtot = sigma(z_pred_total, z_true_total)
eta_tot = eta(z_pred_total, z_true_total) 
err_tot = np.around(np.mean(z_errs_total_cal/(1 + z_true_total)), 5)

plt.figure(figsize=(8, 8))
# plt.scatter(z_true_total, z_pred_total, s=5, c=np.log10(df['SNR_GI']),
#             cmap='jet')
# plt.colorbar()
plt.errorbar(z_true_total, z_pred_total, yerr=z_errs_total_cal, fmt='.', 
             c='red', ecolor='lightblue', mew=0, elinewidth=0.5)
zmin = np.min(z_true_total)
zmax = np.max(z_true_total)
plt.xlim(zmin, zmax)
plt.ylim(zmin, zmax)
a = np.arange(7)
plt.plot(a, a, c='k', alpha=0.5)
plt.plot(a, 1.02 * a + 0.02, 'k--', alpha=0.5)
plt.plot(a, 0.98 * a - 0.02, 'k--', alpha=0.5)
plt.xlabel(r'$z_{\rm true}$', fontsize=16)
plt.ylabel(r'$z_{\rm pred}$', fontsize=16)
# ax.text(0.2, 1.2, r'$\sigma_{\rm NMAD}$ = ' + str(sig), fontsize=14)
# ax.text(0.2, 1.15, r'$\eta$ = ' + str(out) + '%', fontsize=14)
plt.text(0.2, 1.2, r'$\sigma_{\rm NMAD}$ = ' + str(sigtot), fontsize=16)
plt.text(0.2, 1.12, r'$\eta$ = ' + str(eta_tot) + '%', fontsize=16)
plt.text(0.2, 1.04, r'$\overline{E}$ = ' + str(err_tot), fontsize=16)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
figname = os.path.join(result_dir, 'result_after_calibration.png')
plt.savefig(figname)

filename = os.path.join(result_dir, 'result_calibrated.npz')
np.savez_compressed(filename, z_true=z_true_total,
                    z_pred=np.column_stack((z_pred_total, z_errs_total_cal)))