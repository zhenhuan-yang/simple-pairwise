from sklearn.model_selection import RepeatedKFold
from auc_fifo_hinge import auc_fifo_hinge
from auc_olp_hinge import auc_olp_hinge
from auc_oam_gra_hinge import auc_oam_gra_hinge
from auc_olp_1_hinge import auc_olp_1_hinge
from auc_oam_gra_1_hinge import auc_oam_gra_1_hinge
from auc_pair_hinge import auc_pair_hinge
from auc_fifo_loglink import auc_fifo_loglink
from auc_fifo_logistic import auc_fifo_logistic
from auc_spauc_square import auc_spauc_square
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import itertools
from utils import get_res_idx

# cross validation, the parameters are stored in para_cand: eta is the initial step size and beta is the regularization parameter
def exp_cv(x, y, loss, method, options):
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
    for k, auc_ in results:
        if auc_ > auc_max:
            auc_max = auc_
            k_max = k
        #    results = list(results)
        # print('eta=%.4f beta=%.4f auc=%.6f' % (eta_list[k//n_beta], beta_list[k%n_beta],auc_))
    ret['eta_opt'] = eta_list[k_max // n_beta]
    ret['beta_opt'] = beta_list[k_max % n_beta]
    ret['auc_cv'] = auc_max
    print('best eta=%.4f best beta=%.4f auc=%.6f' % (eta_list[k_max // n_beta], beta_list[k_max % n_beta], auc_max))
    return ret


def help_para(arg1, arg2):
    arg = arg1 + (arg2,)
    return auc_fix_para(*arg)


# the auc of the output with a fixed parameter, k is used to indicate the specified eta and beta
def auc_fix_para(x, y, loss, method, options, k):
    num_split = 5
    num_repeat = options['n_repeat']
    k_fold = RepeatedKFold(n_splits=num_split, n_repeats=num_repeat)
    eta_list = options['eta_list']
    beta_list = options['beta_list']
    n_beta = len(beta_list)
    k_eta = k // n_beta
    k_beta = k % n_beta
    options['eta'] = eta_list[k_eta]
    options['beta'] = beta_list[k_beta]
    #    options['n_pass'] = 10   # 8 passes
    auc_s = np.zeros(num_split * num_repeat)

    #    n_split = 0
    for i, (tr_idx, te_idx) in enumerate(k_fold.split(x)):
        n_1 = int(0.8 * len(tr_idx))
        n_2 = int(0.8 * len(te_idx))
        tr_idx = tr_idx[:n_1]
        te_idx = te_idx[:n_2]
        x_tr, x_te = x[tr_idx], x[te_idx]  # we only use 80 percents for training and validation
        y_tr, y_te = y[tr_idx], y[te_idx]
        options['res_idx'] = get_res_idx(options['n_pass'] * (n_1 - 1), options)
        if loss == 'hinge':
            if method == 'auc_fifo':
                auc_t, _ = auc_fifo_hinge(x_tr, y_tr, x_te, y_te, options)
            elif method == 'auc_pair':
                auc_t, _ = auc_pair_hinge(x_tr, y_tr, x_te, y_te, options)
            elif method == 'auc_olp':
                auc_t, _ = auc_olp_hinge(x_tr, y_tr, x_te, y_te, options)
            elif method == 'auc_oam_gra':
                auc_t, _ = auc_oam_gra_hinge(x_tr, y_tr, x_te, y_te, options)
            elif method == 'auc_olp_1':
                auc_t, _ = auc_olp_1_hinge(x_tr, y_tr, x_te, y_te, options)
            elif method == 'auc_oam_gra_1':
                auc_t, _ = auc_oam_gra_1_hinge(x_tr, y_tr, x_te, y_te, options)
            else:
                print('Wrong method name!')
                return
        elif loss == 'loglink':
            if method == 'auc_fifo':
                auc_t, _ = auc_fifo_loglink(x_tr, y_tr, x_te, y_te, options)
            else:
                print('Wrong method name!')
                return
        elif loss == 'logistic':
            if method == 'auc_fifo':
                auc_t, _ = auc_fifo_logistic(x_tr, y_tr, x_te, y_te, options)
            else:
                print('Wrong method name!')
                return
        elif loss == 'square':
            if method == 'auc_spauc':
                auc_t, _ = auc_spauc_square(x_tr, y_tr, x_te, y_te, options)
            else:
                print('Wrong method name!')
                return
        else:
            print('Wrong loss name!')
            return
        auc_s[i] = auc_t[-1]

    return k, np.mean(auc_s)

if __name__ == '__main__':
    from utils import data_processing
    import json
    import os

    options = dict()
    options['n_proc'] = 4
    options['n_pass'] = 10
    options['rec_log'] = .5
    options['rec'] = 50
    options['log_res'] = True
    options['eta_list'] = 10. ** np.arange(-3, 3)
    options['beta_list'] = 10. ** np.arange(-3, 3)
    options['buffer'] = 100
    options['n_repeat'] = 5

    data = 'vehicle'
    loss = 'hinge'
    method = 'auc_fifo'
    x, y = data_processing(data)
    ret = exp_cv(x, y, loss, method, options)

    cur_path = os.getcwd()
    config_path = os.path.join(cur_path, 'config')
    filename = data + '_' + method + '_' + loss

    with open(os.path.join(config_path, filename + '.json'), 'w') as file:
        json.dump(ret, file)

