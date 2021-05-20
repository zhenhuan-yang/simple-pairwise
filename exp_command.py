from exp_cv import exp_cv
from exp_test import exp_test
from utils import data_processing
import numpy as np
import os
import json
import pickle as pkl

def exp_command(data, loss, method):
    options = dict()
    options['n_proc'] = 10
    options['rec_log'] = .5
    options['rec'] = 50
    options['log_res'] = True
    options['eta_list'] = 10. ** np.arange(-3, 3)
    options['beta_list'] = 10. ** np.arange(-3, 3)
    options['buffer'] = 100
    options['n_repeat'] = 5

    x, y = data_processing(data)
    cur_path = os.getcwd()
    config_path = os.path.join(cur_path, 'config')

    if method == 'auc_fifo' or method == 'auc_pair' or method == 'auc_spauc':
        options['n_pass'] = 10
    elif method == 'auc_oam_gra':
        options['n_pass'] = 1
        options['buffer'] = 100
    elif method == 'auc_oam_gra_1':
        options['n_pass'] = 2
        options['buffer'] = 1
    elif method == 'auc_olp_1':
        options['n_pass'] = 2
        options['buffer'] = 2
    else:
        options['n_pass'] = 1
        options['buffer'] = 200
    ret = exp_cv(x, y, loss, method, options)
    filename = data + '_' + method + '_' + loss

    with open(os.path.join(config_path, filename + '.json'), 'w') as file:
        json.dump(ret, file)

    options['eta'] = ret['eta_opt']
    options['beta'] = ret['beta_opt']

    auc, time = exp_test(x, y, loss, method, options)
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
    parser.add_argument('-l', '--loss', default='hinge')
    parser.add_argument('-m', '--method', default='auc_fifo')
    parser.add_argument('-p', '--private', action='store_true')
    args = parser.parse_args()

    exp_command(args.data, args.loss, args.method)