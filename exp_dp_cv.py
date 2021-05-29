from sklearn.model_selection import RepeatedKFold
from alg import auc_dp_fifo_logistic
from alg import auc_dp_egd_logistic
from alg import auc_dp_offpairc_logistic
from alg import auc_fifo_logistic
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import itertools
from utils import get_stage_idx, get_stage_res_idx, get_etas


# cross validation, the parameters are stored in para_cand: eta is the initial step size and beta is the regularization parameter
def exp_dp_cv(x, y, loss, method, options):
    ret = dict()
    eta_list = options['eta_list']
    beta_list = options['beta_list']
    n_beta = len(beta_list)
    n_cv = len(eta_list) * n_beta
    #    pool = mp.Pool(processes=para_cand['n_proc'])
    #    results = [pool.apply(auc_fix_para, args=(x, y, method, para_cand, n_pass, k)) for k in range(n_cv)]
    #    with ProcessPoolExecutor(para_cand['n_proc']) as executor:
    #        args = ((x, y, method, para_cand, n_pass, k) for k in range(n_cv))
    #        print(args)
    #        results = executor.map(lambda p: auc_fix_para(*p), args)

    cv_set = range(n_cv)
    with ProcessPoolExecutor(options['n_proc']) as executor:
        results = executor.map(help_para, itertools.repeat((x, y, loss, method, options)), cv_set)
    auc_max, std_max, k_max = 0, 0, 0
    for k, auc_mean, auc_std in results:
        if auc_mean > auc_max:
            auc_max = auc_mean
            k_max = k
            std_max = auc_std
        #    results = list(results)
        # print('eta=%.4f beta=%.4f auc=%.6f std=%.6f' % (eta_list[k//n_beta], beta_list[k%n_beta],auc_mean,auc_std))
    ret['eta_opt'] = eta_list[k_max // n_beta]
    ret['beta_opt'] = beta_list[k_max % n_beta]
    ret['auc_cv'] = auc_max
    ret['std_cv'] = std_max
    ret['epsilon'] = options['epsilon']
    ret['delta'] = options['delta']
    print('eps=%.1f best eta=%.4f best beta=%.4f auc=%.6f std=%.6f'
          % (options['epsilon'], eta_list[k_max // n_beta], beta_list[k_max % n_beta], auc_max, std_max))
    return ret


def help_para(arg1, arg2):
    arg = arg1 + (arg2,)
    return auc_fix_para(*arg)


# the auc of the output with a fixed parameter, k is used to indicate the specified eta and beta
def auc_fix_para(x, y, loss, method, options, k):
    stages = get_stage_idx(options['n_tr'])
    options['res_idx'] = get_stage_res_idx(stages, options['n_pass'])
    n = len(y)
    num_split = options['n_split']
    # if num_split < 2:
    #     num_split = 2
    num_repeat = options['n_repeat']
    k_fold = RepeatedKFold(n_splits=num_split, n_repeats=num_repeat)
    eta_list = options['eta_list']
    beta_list = options['beta_list']
    n_beta = len(beta_list)
    k_eta = k // n_beta
    k_beta = k % n_beta
    options['eta'] = eta_list[k_eta]
    options['beta'] = beta_list[k_beta]
    options['etas'] = get_etas(options['n_pass'] * options['n_tr'], options['eta'], options)
    #    options['n_pass'] = 10   # 8 passes
    auc_s = np.zeros(num_split * num_repeat)

    #    n_split = 0
    for i, (tr_idx, te_idx) in enumerate(k_fold.split(x)):
        tr_idx = tr_idx[:options['n_tr']]
        # te_idx = te_idx[:n_2]
        x_tr, x_te = x[tr_idx], x[te_idx]  # we only use 80 percents for training and validation
        y_tr, y_te = y[tr_idx], y[te_idx]
        # small training!
        # x_tr, x_te = x[te_idx], x[tr_idx]
        # y_tr, y_te = y[te_idx], y[tr_idx]
        if loss == 'logistic':
            if method == 'auc_dp_fifo':
                auc_t, _ = auc_dp_fifo_logistic(x_tr, y_tr, x_te, y_te, options)
            elif method == 'auc_dp_egd':
                auc_t, _ = auc_dp_egd_logistic(x_tr, y_tr, x_te, y_te, options)
            elif method == 'auc_dp_offpairc':
                auc_t, _ = auc_dp_offpairc_logistic(x_tr, y_tr, x_te, y_te, options)
            elif method == 'auc_fifo':
                auc_t, _ = auc_fifo_logistic(x_tr, y_tr, x_te, y_te, options)
            else:
                print('Wrong method name!')
                return
        else:
            print('Wrong loss name!')
            return
        auc_s[i] = auc_t[-1]

    return k, np.mean(auc_s), np.std(auc_s)

if __name__ == '__main__':
    from utils import data_processing, get_stage_idx, get_stage_res_idx
    import json
    import os

    options = dict()
    options['n_proc'] = 4
    options['rec_log'] = .5
    options['eta_list'] = 10. ** np.arange(-3, 1)
    options['beta_list'] = 10. ** np.arange(-3, 3)
    # options['batch_size'] = 128
    options['epsilon'] = 1.
    options['n_tr'] = 256
    options['n_pass'] = 10
    options['proj_flag'] = True
    stages = get_stage_idx(options['n_tr'])
    options['res_idx'] = get_stage_res_idx(stages, options['n_pass'])
    options['n_repeat'] = 2
    options['delta'] = 1. / options['n_tr']
    data = 'diabetes'
    loss = 'logistic'
    method = 'auc_fifo'

    x, y = data_processing(data)
    n, options['dim'] = x.shape
    options['n_split'] = int(n / options['n_tr'])

    # k, auc_mean, auc_std = auc_fix_para(x, y, loss, method, options, 1)
    ret = exp_dp_cv(x, y, loss, method, options)

    cur_path = os.getcwd()
    config_path = os.path.join(cur_path, 'config')
    filename = data + '_' + method + '_' + loss + '_eps_' + str(options['epsilon'])

    with open(os.path.join(config_path, filename + '.json'), 'w') as file:
        json.dump(ret, file)

