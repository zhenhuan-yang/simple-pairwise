from alg import auc_fifo_hinge
from alg import auc_fifo_square
from alg import auc_fifo_loglink
from alg import auc_spauc_square
from alg import auc_fifo_logistic
from alg import auc_olp_hinge
from alg import auc_olp_square
from alg import auc_oam_gra_hinge
from alg import auc_olp_1_hinge
from alg import auc_oam_gra_1_hinge
from alg import auc_pair_hinge
from alg import auc_pair_square
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from sklearn.model_selection import RepeatedKFold
import itertools
from utils import get_res_idx, get_etas

def exp_test(x, y, loss, method, options):
    options['res_idx'] = get_res_idx(options['n_pass'] * (options['n_tr'] - 1), options)
    # unify the record length
    # n_tr - 1 to avoid overflowing
    n_idx = len(options['res_idx'])
    # define the etas to run all test since unified
    options['etas'] = get_etas(options['n_pass'] * options['n_tr'], options['eta'], options)
    time_rec = np.zeros([options['n_split'] * options['n_repeat'], n_idx])
    auc_rec = np.zeros([options['n_split'] * options['n_repeat'], n_idx])
    k = 0
    k_fold = RepeatedKFold(n_splits=options['n_split'], n_repeats=options['n_repeat'])
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
    print('method: %s auc: $%s \pm %s$' % (method, mean_str, std_str))
    return auc, time

def help_para(arg1, arg2):
    arg = arg1 + (arg2,)
    return auc_fix_para(*arg)


# the auc of the output with a fixed parameter, k is used to indicate the specified eta and beta
def auc_fix_para(x, y, loss, method, options, idx):
    idx_tr = idx[0]
    idx_te = idx[1]
    x_tr, x_te = x[idx_tr[:options['n_tr']]], x[idx_te]
    y_tr, y_te = y[idx_tr[:options['n_tr']]], y[idx_te]

    if loss == 'hinge':
        if method == 'auc_fifo':
            auc_t, time_t = auc_fifo_hinge(x_tr, y_tr, x_te, y_te, options)
        elif method == 'auc_olp':
            auc_t, time_t = auc_olp_hinge(x_tr, y_tr, x_te, y_te, options)
        elif method == 'auc_oam_gra':
            auc_t, time_t = auc_oam_gra_hinge(x_tr, y_tr, x_te, y_te, options)
        elif method == 'auc_olp_1':
            auc_t, time_t = auc_olp_1_hinge(x_tr, y_tr, x_te, y_te, options)
        elif method == 'auc_oam_gra_1':
            auc_t, time_t = auc_oam_gra_1_hinge(x_tr, y_tr, x_te, y_te, options)
        elif method == 'auc_pair':
            auc_t, time_t = auc_pair_hinge(x_tr, y_tr, x_te, y_te, options)
        else:
            print('Wrong method name!')
            return
    elif loss == 'loglink':
        if method == 'auc_fifo':
            auc_t, time_t = auc_fifo_loglink(x_tr, y_tr, x_te, y_te, options)
        else:
            print('Wrong method name!')
            return
    elif loss == 'logistic':
        if method == 'auc_fifo':
            auc_t, time_t = auc_fifo_logistic(x_tr, y_tr, x_te, y_te, options)
        else:
            print('Wrong method name!')
            return
    elif loss == 'square':
        if method == 'auc_spauc':
            auc_t, time_t = auc_spauc_square(x_tr, y_tr, x_te, y_te, options)
        elif method == 'auc_fifo':
            auc_t, time_t = auc_fifo_square(x_tr, y_tr, x_te, y_te, options)
        elif method == 'auc_pair':
            auc_t, time_t = auc_pair_square(x_tr, y_tr, x_te, y_te, options)
        elif method == 'auc_olp':
            auc_t, time_t = auc_olp_square(x_tr, y_tr, x_te, y_te, options)
        else:
            print('Wrong method name!')
            return
    else:
        print('Wrong loss name!')
        return
    return auc_t, time_t

if __name__ == '__main__':
    from utils import data_processing, get_idx
    import json
    import pickle as pkl
    import os
    import timeit

    options = dict()
    options['n_proc'] = 4
    options['n_pass'] = 10
    options['rec_log'] = .5
    options['rec'] = 100
    options['log_res'] = True
    options['buffer'] = 200
    options['n_repeat'] = 5
    # options['n_split'] = 5
    options['proj_flag'] = True
    options['eta_geo'] = 'const'

    data = 'diabetes'
    loss = 'hinge'
    method = 'auc_fifo'
    x, y = data_processing(data)
    n, options['dim'] = x.shape
    # unify number of training
    options['n_split'] = 5
    options['n_tr'] = int((options['n_split'] - 1) / options['n_split'] * n)
    # options['n_tr'] = 256
    # options['n_split'] = int(n / options['n_tr'])

    cur_path = os.getcwd()
    config_path = os.path.join(cur_path, 'config')

    res_path = os.path.join(cur_path, 'res')
    filename = data + '_' + method + '_' + loss # + '_4dp'

    if os.path.exists(os.path.join(config_path, filename + '.json')):
        with open(os.path.join(config_path, filename + '.json'), 'r') as file:
            ret = json.load(file)
            options['eta'] = ret['eta_opt']
            options['beta'] = ret['beta_opt']
    else:
        options['eta'] = 1e2
        options['beta'] = 1e2

    auc, time = exp_test(x, y, loss, method, options)
    res = dict()
    res['auc'] = auc
    res['time'] = time
    with open(os.path.join(res_path, filename + '.pickle'), 'wb') as pklfile:
        pkl.dump(res, pklfile)