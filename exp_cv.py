from sklearn.model_selection import RepeatedKFold
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
import itertools
from utils import get_res_idx, get_etas

# cross validation, the parameters are stored in para_cand: eta is the initial step size and beta is the regularization parameter
def exp_cv(x, y, loss, method, options):
    options['res_idx'] = get_res_idx(options['n_pass'] * (options['n_tr'] - 1), options)
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
    auc_max, k_max = 0, 0
    time_max = 0.
    for k, auc_, time_ in results:
        if auc_ > auc_max:
            auc_max = auc_
            time_max = time_
            k_max = k
        #    results = list(results)
        # print('eta=%.4f beta=%.4f auc=%.6f' % (eta_list[k//n_beta], beta_list[k%n_beta],auc_))
    ret['eta_opt'] = eta_list[k_max // n_beta]
    ret['beta_opt'] = beta_list[k_max % n_beta]
    ret['auc_cv'] = auc_max
    ret['time_elapsed'] = time_max
    ret['n_pass'] = options['n_pass']
    print('best eta=%.4f best beta=%.4f auc=%.6f, time=%.6f' % (eta_list[k_max // n_beta], beta_list[k_max % n_beta], auc_max, time_max))
    return ret


def help_para(arg1, arg2):
    arg = arg1 + (arg2,)
    return auc_fix_para(*arg)


# the auc of the output with a fixed parameter, k is used to indicate the specified eta and beta
def auc_fix_para(x, y, loss, method, options, k):
    k_fold = RepeatedKFold(n_splits=options['n_split'], n_repeats=options['n_repeat'])
    eta_list = options['eta_list']
    beta_list = options['beta_list']
    n_beta = len(beta_list)
    k_eta = k // n_beta
    k_beta = k % n_beta
    options['eta'] = eta_list[k_eta]
    options['beta'] = beta_list[k_beta]
    # define etas for this particular eta
    options['etas'] = get_etas(options['n_pass'] * options['n_tr'], options['eta'], options) #options will decide eta_geo
    #    options['n_pass'] = 10   # 8 passes
    auc_s = np.zeros(options['n_split'] * options['n_repeat'])
    time_s = np.zeros(options['n_split'] * options['n_repeat'])
    #    n_split = 0
    for i, (tr_idx, te_idx) in enumerate(k_fold.split(x)):
        # n_1 = int(0.8 * len(tr_idx))
        # n_2 = int(0.8 * len(te_idx))
        tr_idx = tr_idx[:options['n_tr']]
        # te_idx = te_idx[:n_2]
        x_tr, x_te = x[tr_idx], x[te_idx]  # we only use 80 percents for training and validation
        y_tr, y_te = y[tr_idx], y[te_idx]
        if loss == 'hinge':
            if method == 'auc_fifo':
                auc_t, time_t = auc_fifo_hinge(x_tr, y_tr, x_te, y_te, options)
            elif method == 'auc_pair':
                auc_t, time_t = auc_pair_hinge(x_tr, y_tr, x_te, y_te, options)
            elif method == 'auc_olp':
                auc_t, time_t = auc_olp_hinge(x_tr, y_tr, x_te, y_te, options)
            elif method == 'auc_oam_gra':
                auc_t, time_t = auc_oam_gra_hinge(x_tr, y_tr, x_te, y_te, options)
            elif method == 'auc_olp_1':
                auc_t, time_t = auc_olp_1_hinge(x_tr, y_tr, x_te, y_te, options)
            elif method == 'auc_oam_gra_1':
                auc_t, time_t = auc_oam_gra_1_hinge(x_tr, y_tr, x_te, y_te, options)
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
            elif method == 'auc_olp':
                auc_t, time_t = auc_olp_square(x_tr, y_tr, x_te, y_te, options)
            elif method == 'auc_pair':
                auc_t, time_t = auc_pair_square(x_tr, y_tr, x_te, y_te, options)
            else:
                print('Wrong method name!')
                return
        else:
            print('Wrong loss name!')
            return
        # auc_s[i] = auc_t[-1]
        auc_s[i] = np.max(auc_t)
        time_s[i] = time_t[-1]
    return k, np.mean(auc_s), np.mean(time_s)

if __name__ == '__main__':
    from utils import data_processing, get_idx
    import json
    import os
    import timeit

    options = dict()
    options['n_proc'] = 4
    options['n_pass'] = 10
    options['rec_log'] = .5
    options['rec'] = 50
    options['log_res'] = True
    options['eta_list'] = 10. ** np.arange(-3, 3)
    options['beta_list'] = 10. ** np.arange(-3, 3)
    options['buffer'] = 100
    options['n_repeat'] = 2
    options['n_split'] = 5
    options['proj_flag'] = True
    options['eta_geo'] = 'const'
    data = 'diabetes'
    loss = 'hinge'
    method = 'auc_fifo'
    x, y = data_processing(data)
    n, options['dim'] = x.shape
    # start = timeit.default_timer()
    # unify the number of training
    options['n_tr'] = int((options['n_split'] - 1) / options['n_split'] * n)
    # options['n_tr'] = 500
    # options['n_split'] = int(n / options['n_tr'])

    ret = exp_cv(x, y, loss, method, options)

    cur_path = os.getcwd()
    config_path = os.path.join(cur_path, 'config')
    filename = data + '_' + method + '_' + loss # + '_4dp'

    with open(os.path.join(config_path, filename + '.json'), 'w') as file:
        json.dump(ret, file)

