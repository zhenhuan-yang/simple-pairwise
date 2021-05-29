from exp_dp_cv import exp_dp_cv
from exp_dp_test import exp_dp_test
from utils import data_processing, get_stage_idx, get_stage_res_idx
import numpy as np
import os
import json
import pickle as pkl

def exp_dp_command(data, loss, method, epsilon):
    options = dict()
    options['n_proc'] = 12
    options['rec_log'] = .5
    options['rec'] = 50
    options['log_res'] = True
    options['n_repeat'] = 15
    options['epsilon'] = epsilon
    options['n_tr'] = 256
    options['delta'] = 1. / options['n_tr']
    options['proj_flag'] = False
    options['eta_geo'] = 'const'

    x, y = data_processing(data)
    n, options['dim'] = x.shape
    # unify number of training
    options['n_split'] = int(n / options['n_tr'])

    cur_path = os.getcwd()
    config_path = os.path.join(cur_path, 'config')

    if method == 'egd':
        options['n_pass'] = int(options['n_tr'])
        options['eta_list'] = 10. ** np.arange(-3, 1)
        options['beta_list'] = 10. ** np.arange(-3, 1)
        method_name = 'auc_dp_' + method
        filename = data + '_' + method_name + '_' + loss + '_eps_' + str(options['epsilon'])
    elif method == 'fifo':
        options['n_pass'] = 20
        options['eta_list'] = 10. ** np.arange(-6, -2)
        options['beta_list'] = 10. ** np.arange(-4, 2)
        method_name = 'auc_dp_' + method
        filename = data + '_' + method_name + '_' + loss + '_eps_' + str(options['epsilon'])
    elif method == 'offpairc':
        options['n_pass'] = 2
        options['eta_list'] = 10. ** np.arange(-1, 3)
        options['beta_list'] = 10. ** np.arange(-3, 3)
        method_name = 'auc_dp_' + method
        filename = data + '_' + method_name + '_' + loss + '_eps_' + str(options['epsilon'])
    elif method == 'non-private':
        options['n_pass'] = 3
        options['eta_list'] = 10. ** np.arange(-3, 3)
        options['beta_list'] = 10. ** np.arange(-3, 3)
        method_name = 'auc_fifo'
        filename = data + '_' + method_name + '_' + loss

    ret = exp_dp_cv(x, y, loss, method_name, options)
    with open(os.path.join(config_path, filename + '.json'), 'w') as file:
        json.dump(ret, file)

    options['eta'] = ret['eta_opt']
    options['beta'] = ret['beta_opt']

    auc, time = exp_dp_test(x, y, loss, method_name, options)
    res = dict()
    res['auc'] = auc
    res['time'] = time
    res_path = os.path.join(cur_path, 'res')
    with open(os.path.join(res_path, filename + '.pickle'), 'wb') as pklfile:
        pkl.dump(res, pklfile)

    return

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Training setting')
    parser.add_argument('-d', '--data', default='diabetes')
    parser.add_argument('-l', '--loss', default='logistic')
    parser.add_argument('-m', '--method', default='fifo')
    parser.add_argument('-e', '--epsilon', type=float, default=1., help='must be float')
    parser.add_argument('-n', '--n_tr', default=256, type=int,  help='must be int')
    parser.add_argument('-p', '--non_private', action='store_true')
    args = parser.parse_args()

    exp_dp_command(args.data, args.loss, args.method, args.epsilon, args.n_tr)