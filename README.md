# CSST slitless spectra
Relevant codes for "Accurately Estimating Redshifts from CSST Slitless Spectroscopic Survey using Deep Learning" at [arXiv:2407.13991](https://arxiv.org/abs/2407.13991).

## Slitless spectra simulation software
The codes for this software is publicly available at [here](https://csst-tb.bao.ac.cn/code/zhangxin/sls_1d_spec)

## Data
The CSST slitless spectra dataset used for training, validation and testing of our work is publicly available at [here](https://pan.cstcloud.cn/s/E6FrFGa6TJA)

## Usage

### Dependence
`numpy`

`pandas` 

`scipy` 

`joblib` 

`matplotlib` 

Our codes are run on tensorflow versions as follows:

`tensorflow==2.8.0`

`tensorflow-probability==0.16.0`

`keras==2.8.0`

Modifications are possibly needed if use other versions. 

### CNN backbone
`python train_sls.py --model_type=CNN --batch_size=1024 --epochs=100 --augmentation --num_augments=50`
where `--augmentation` means augment data using Gaussian realizations and `num_augments` is the number of augmented data.

### BNN
`python train_sls.py --model_type=BNN --batch_size=1024 --epochs=100 --augmentation --num_augments=50 --n_runs=200 --transfer`
where `n_runs` indicates the number of runs when feeding testing data to BNN and `transfer` means transfer learning is applied using CNN backbone.

If you want to train the BNN from scratch, just delete the `--transfer`.

## Calibration
`python calibration.py --result_file=BNN_1d/result.npz`

