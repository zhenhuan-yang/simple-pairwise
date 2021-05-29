from alg import auc_dp_fifo_logistic
from alg import auc_dp_egd_logistic
from alg import auc_dp_offpairc_logistic
from alg import auc_fifo_logistic
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from sklearn.model_selection import RepeatedKFold
import itertools
from utils import get_stage_idx, get_stage_res_idx, get_etas

def exp_dp_test(x, y, loss, method, options):
    n = len(y)
    num_split = options['n_split']
    # if num_split < 2:
    #     num_split = 2
    num_repeat = options['n_repeat']
    # unify the record length
    # n_tr = int(len(y) / num_split)
    # n_tr - 1 to avoid overflowing
    stages = get_stage_idx(options['n_tr'])
    options['res_idx'] = get_stage_res_idx(stages, options['n_pass'])
    options['etas'] = get_etas(options['n_pass'] * options['n_tr'], options['eta'], options)
    n_idx = len(options['res_idx'])
    time_rec = np.zeros([num_split * num_repeat, n_idx])
    auc_rec = np.zeros([num_split * num_repeat, n_idx])
    k = 0
    k_fold = RepeatedKFold(n_splits=num_split, n_repeats=num_repeat)
    with ProcessPoolExecutor(options['n_proc']) as executor:
        results = executor.map(help_para, itertools.repeat((x, y, loss, method, options)), k_fold.split(x))
    for auc_, time_ in results:
        time_rec[k] = time_
        auc_rec[k] = auc_
        k = k + 1
    time = dict()
    auc = dict()
    time['mean'] = np.mean(time_rec, 0)
    time['std'] = np.std(time_rec, 0)
    auc['mean'] = np.mean(auc_rec, 0)
    auc['std'] = np.std(auc_rec, 0)

    mean_str = str(np.max(auc['mean'])).lstrip('0')[:4]
    std_str = str(np.min(auc['std'])).lstrip('0')[:4]
    print('method: %s n_tr: %d eps: %.1f auc: $%s \pm %s$' % (method, options['n_tr'], options['epsilon'], mean_str, std_str))
    return auc, time

def help_para(arg1, arg2):
    arg = arg1 + (arg2,)
    return auc_fix_para(*arg)


# the auc of the output with a fixed parameter, k is used to indicate the specified eta and beta
def auc_fix_para(x, y, loss, method, options, idx):
    # smaller for training
    # idx_tr = idx[1]
    # idx_te = idx[0]
    # x_tr, x_te = x[idx_tr], x[idx_te]
    # y_tr, y_te = y[idx_tr], y[idx_te]
    idx_tr = idx[0]
    idx_te = idx[1]
    x_tr, x_te = x[idx_tr[:options['n_tr']]], x[idx_te]
    y_tr, y_te = y[idx_tr[:options['n_tr']]], y[idx_te]

    if loss == 'logistic':
        if method == 'auc_dp_fifo':
            auc_t, time_t = auc_dp_fifo_logistic(x_tr, y_tr, x_te, y_te, options)
        elif method == 'auc_dp_egd':
            auc_t, time_t = auc_dp_egd_logistic(x_tr, y_tr, x_te, y_te, options)
        elif method == 'auc_dp_offpairc':
            auc_t, time_t = auc_dp_offpairc_logistic(x_tr, y_tr, x_te, y_te, options)
        elif method == 'auc_fifo':
            auc_t, time_t = auc_fifo_logistic(x_tr, y_tr, x_te, y_te, options)
        else:
            print('Wrong method name!')
    else:
        print('Wrong loss name!')
        return
    return auc_t, time_t

if __name__ == '__main__':
    from utils import data_processing
    import json
    import pickle as pkl
    import os

    options = dict()
    options['n_proc'] = 12
    options['n_pass'] = 10
    options['n_tr'] = 256
    stages = get_stage_idx(options['n_tr'])
    options['res_idx'] = get_stage_res_idx(stages, options['n_pass'])
    options['epsilon'] = 1.
    options['proj_flag'] = True

    options['n_repeat'] = 10
    options['delta'] = 1. / options['n_tr']

    data = 'diabetes'
    loss = 'logistic'
    method = 'auc_fifo'
    x, y = data_processing(data)
    n, options['dim'] = x.shape
    options['n_split'] = int(n / options['n_tr'])

    cur_path = os.getcwd()
    config_path = os.path.join(cur_path, 'config')

    res_path = os.path.join(cur_path, 'res')
    filename = data + '_' + method + '_' + loss + '_eps_' + str(options['epsilon'])

    if os.path.exists(os.path.join(config_path, filename + '.json')):
        with open(os.path.join(config_path, filename + '.json'), 'r') as file:
            ret = json.load(file)
            options['eta'] = ret['eta_opt']
            options['beta'] = ret['beta_opt']
    else:
        print('config file not found!')
        options['eta'] = 1e0
        options['beta'] = 1e0

    auc, time = exp_dp_test(x, y, loss, method, options)
    res = dict()
    res['auc'] = auc
    res['time'] = time
    with open(os.path.join(res_path, filename + '.pickle'), 'wb') as pklfile:
        pkl.dump(res, pklfile)