from exp_cv import exp_cv
from exp_test import exp_test
from utils import data_processing, get_idx, get_pair_idx
import numpy as np
import os
import json
import pickle as pkl

def plot_exp_command(data, loss, method):
    options = dict()
    options['n_proc'] = 12
    options['rec_log'] = .5
    options['rec'] = 50
    options['log_res'] = True
    options['eta_list'] = 10. ** np.arange(-3, 3)
    options['beta_list'] = [100.] # only one beta
    options['n_repeat'] = 5
    options['n_split'] = 5

    x, y = data_processing(data)
    n, options['dim'] = x.shape

    # unify number of training
    options['n_tr'] = int((options['n_split'] - 1) / options['n_split'] * n)

    # unify constant step size and require projection
    options['eta_geo'] = 'const' # fix step sizes
    options['proj_flag'] = True

    cur_path = os.getcwd()
    config_path = os.path.join(cur_path, 'config')

    if method == 'fifo':
        options['n_pass'] = 20
        # start = timeit.default_timer()
        # options['ids'] = get_idx(options['n_tr'], options['n_pass'])
        # stop = timeit.default_timer()
        # options['time_avg'] = (stop - start) / (options['n_pass'] * options['n_tr'])
    elif method == 'pair':
        options['n_pass'] = 20
        # start = timeit.default_timer()
        # options['ids'] = get_pair_idx(options['n_tr'], options['n_pass'])
        # stop = timeit.default_timer()
        # options['time_avg'] = (stop - start) / (options['n_pass'] * options['n_tr'])
    elif method == 'oam_gra':
        options['n_pass'] = 5
        options['buffer'] = 100
        options['proj_flag'] = False
        # start = timeit.default_timer()
        # options['ids'] = get_idx(options['n_tr'], options['n_pass'])
        # stop = timeit.default_timer()
        # options['time_avg'] = (stop - start) / (options['n_pass'] * options['n_tr'])
    elif method == 'oam_gra_1' or method == 'olp_1':
        options['n_pass'] = 20
        options['buffer'] = 1
        # start = timeit.default_timer()
        # options['ids'] = get_idx(options['n_tr'], options['n_pass'])
        # stop = timeit.default_timer()
        # options['time_avg'] = (stop - start) / (options['n_pass'] * options['n_tr'])
    elif method == 'olp':
        options['n_pass'] = 5
        options['buffer'] = 200
        # start = timeit.default_timer()
        # options['ids'] = get_idx(options['n_tr'], options['n_pass'])
        # stop = timeit.default_timer()
        # options['time_avg'] = (stop - start) / (options['n_pass'] * options['n_tr'])
    elif method == 'spauc':
        options['n_pass'] = 20
        # start = timeit.default_timer()
        # options['ids'] = get_idx(options['n_tr'], options['n_pass'])
        # stop = timeit.default_timer()
        # options['time_avg'] = (stop - start) / (options['n_pass'] * options['n_tr'])

    method_name = 'auc_' + method
    ret = exp_cv(x, y, loss, method_name, options)
    filename = data + '_' + method_name + '_' + loss + '_4plot'

    with open(os.path.join(config_path, filename + '.json'), 'w') as file:
        json.dump(ret, file)

    options['eta'] = ret['eta_opt']
    options['beta'] = ret['beta_opt']

    auc, time = exp_test(x, y, loss, method_name, options)
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
    parser.add_argument('-m', '--method', default='fifo')
    parser.add_argument('-p', '--private', action='store_true')
    args = parser.parse_args()

    plot_exp_command(args.data, args.loss, args.method)