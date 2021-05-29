import numpy as np
from sklearn import metrics
import timeit
from utils import get_idx, get_stage_idx, get_stage_res_idx

def sigma_calibrator(epsilon, delta, n_tr):
    sigma = 4. * np.sqrt(np.log(1. / delta)) / (np.sqrt(n_tr) * epsilon)
    return sigma

def auc_dp_offpairc_logistic(x_tr, y_tr, x_te, y_te, options):
    n_tr = options['n_tr']
    dim = options['dim']
    n_pass = options['n_pass']
    start = timeit.default_timer()
    ids = get_idx(n_tr, n_pass)
    stop = timeit.default_timer()
    n_iter = len(ids)
    time_avg = (stop - start) / n_iter
    stages = get_stage_idx(n_tr)
    res_idx = get_stage_res_idx(stages, n_pass)
    w_t = np.zeros(dim) # warm start
    w_sum = np.zeros(dim)
    w_sum_old = np.zeros(dim)
    eta_sum = 0.
    eta_sum_old = 0.
    eta = options['eta'] # eta for this partition
    etas = eta / np.sqrt(np.arange(1, n_iter + 1))
    beta = options['beta']
    # idx_pos = y_tr > 0.
    # idx_neg = y_tr < 0.
    # n_pos = np.sum(idx_pos)
    # n_neg = np.sum(idx_neg)
    # if n_pos == 0 or n_neg == 0:
    #     return w_t
    # x_pos = x_tr[idx_pos]
    # x_neg = x_tr[idx_neg]
    sigma = sigma_calibrator(options['epsilon'], options['delta'], n_tr)
    aucs = np.zeros(len(res_idx))
    times = np.zeros(len(res_idx))
    i_res = 0
    t = 2
    time_sum = 0.

    while t < n_iter:
        start = timeit.default_timer()
        # current example
        x_t = x_tr[ids[t]]
        y_t = y_tr[ids[t]]
        eta_t = etas[t]
        # past example
        x_t_1 = x_tr[ids[:t - 1]]
        y_t_1 = y_tr[ids[:t - 1]]
        if y_t > 0.:
            idx_neg = y_t_1 < 0
            n_neg = np.sum(idx_neg)
            if n_neg == 0:
                xx = np.zeros((1, dim))
            else:
                xx = x_t - x_t_1[idx_neg]
        else:
            idx_pos = y_t_1 > 0
            n_pos = np.sum(idx_pos)
            if n_pos == 0:
                xx = np.zeros((1, dim))
            else:
                xx = x_t_1[idx_pos] - x_t
        wxx = np.dot(xx, w_t)
        # avoid overflow
        # mask = np.abs(wxx) > 100.
        # sign = np.sign(wxx)
        # wxx[mask] = sign[mask] * 100.
        gd = - xx * np.exp(-wxx)[:, None] / (1. + np.exp(-wxx)[:, None])
        # gradient step
        w_t = w_t - eta_t * (np.sum(gd, axis=0) / (t - 1) + w_t / np.sqrt(n_tr))
        if options['proj_flag']:
            # projection
            norm = np.linalg.norm(w_t)
            if norm > beta:
                w_t = w_t * beta / norm
        stop = timeit.default_timer()
        time_sum += stop - start
        # naive average
        w_sum = w_sum + w_t
        eta_sum = eta_sum + 1
        # udpate
        t = t + 1
        if i_res < len(res_idx) and (res_idx[i_res] == 1 or res_idx[i_res] == t):
            # average output
            w_avg = (w_sum - w_sum_old) / (eta_sum - eta_sum_old)  # trick: only average between two i_res
            b_t = np.random.normal(0., sigma, dim)
            w_priv = w_avg + b_t
            pred = (x_te.dot(w_priv.T)).ravel()
            if not np.all(np.isfinite(pred)):
                break
            fpr, tpr, thresholds = metrics.roc_curve(y_te, pred.T, pos_label=1)
            aucs[i_res] = metrics.auc(fpr, tpr)
            times[i_res] = time_sum
            i_res = i_res + 1
    return aucs, times


if __name__ == '__main__':
    from utils import data_processing, get_res_idx, get_stage_res_idx

    from sklearn.model_selection import train_test_split, RepeatedKFold

    options = dict()
    options['rec_log'] = .5
    options['rec'] = 50
    options['log_res'] = False
    options['eta'] = 1.
    options['beta'] = 10.
    options['epsilon'] = 2.
    options['n_pass'] = 5
    options['n_tr'] = 256
    options['n_repeat'] = 5
    options['delta'] = 1. / options['n_tr']
    options['proj_flag'] = True
    x, y = data_processing('diabetes')
    n, options['dim'] = x.shape
    num_split = int(n / options['n_tr'])
    stages = get_stage_idx(options['n_tr'])
    res_idx = get_stage_res_idx(stages, options['n_pass'])
    n_idx = len(res_idx)
    time_rec = np.zeros([num_split * options['n_repeat'], n_idx])
    auc_rec = np.zeros([num_split * options['n_repeat'], n_idx])
    k_fold = RepeatedKFold(n_splits=num_split, n_repeats=options['n_repeat'])
    # x_tr, x_te, y_tr, y_te = train_test_split(x, y, train_size=0.25)
    for i, (tr_idx, te_idx) in enumerate(k_fold.split(x)):
        x_tr, x_te = x[te_idx], x[tr_idx]
        y_tr, y_te = y[te_idx], y[tr_idx]
        auc_t, time_t = auc_dp_offpairc_logistic(x_tr, y_tr, x_te, y_te, options)
        time_rec[i] = time_t
        auc_rec[i] = auc_t
    print('----------------------')
    print(np.mean(auc_rec, axis=0))
    print(np.std(auc_rec, axis=0))