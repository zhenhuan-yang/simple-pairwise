import numpy as np
from sklearn import metrics
import timeit
from utils import get_idx

def auc_fifo_loglink(x_tr, y_tr, x_te, y_te, options):
    # get
    n_tr = options['n_tr']
    dim = options['dim']
    n_pass = options['n_pass']
    start = timeit.default_timer()
    ids = get_idx(n_tr, n_pass)
    n_iter = len(ids)
    stop = timeit.default_timer()
    time_avg = (stop - start) / n_iter
    # ids = options['ids']
    # time_avg = options['time_avg']
    res_idx = options['res_idx']
    w_t = np.zeros(dim)
    # w_t_1 = np.zeros(dim)
    # w_t_2 = np.zeros(dim)
    w_sum = np.zeros(dim)
    w_sum_old = np.zeros(dim)
    eta_sum = 0.
    eta_sum_old = 0.
    eta = options['eta'] # initial eta
    etas = options['etas']
    beta = options['beta']
    aucs = np.zeros(len(res_idx))
    times = np.zeros(len(res_idx))
    t = 0
    i_res = 0
    time_sum = 0.
    # initiate timer
    start = timeit.default_timer()
    while t < n_iter:
        # current example
        y_t = y_tr[ids[t]]
        # past example
        y_t_1 = y_tr[ids[t-1]]
        # when ys are different
        if y_t * y_t_1 < 0:
            x_t = x_tr[ids[t]]
            x_t_1 = x_tr[ids[t - 1]]  # start with -1, it suppose to be random
            eta_t = etas[t]
            # make sure positive is in front
            yxx = y_t * (x_t - x_t_1)
            wyxx = np.inner(w_t, yxx)
            # loglink loss gradient
            gd = -2. * yxx * np.exp(-2. * wyxx) / (1. + np.exp(-wyxx))**3
            # gradient step
            w_t = w_t - eta_t * gd
            # w_t = w_t_1 - eta_t * gd
            if options['proj_flag']:
                # projection
                norm = np.linalg.norm(w_t)
                if norm > beta:
                    w_t = w_t * beta / norm
        # naive average
        w_sum = w_sum + w_t
        eta_sum = eta_sum + 1
        # # average eta with two step ago
        # w_sum = w_sum + eta_t * w_t_2
        # eta_sum = eta_sum + eta_t
        # # move t-2
        # w_t_2 = w_t_1 + 0.
        # # move t-1
        # w_t_1 = w_t + 0.
        # update
        t = t + 1
        # save results
        if i_res < len(res_idx) and res_idx[i_res] == t:
            # stop timer
            stop = timeit.default_timer()
            time_sum += stop - start + time_avg * (eta_sum - eta_sum_old)
            # average output
            w_avg = (w_sum - w_sum_old) / (eta_sum - eta_sum_old) # trick: only average between two i_res
            pred = (x_te.dot(w_avg.T)).ravel()
            if not np.all(np.isfinite(pred)):
                break
            fpr, tpr, thresholds = metrics.roc_curve(y_te, pred.T, pos_label=1)
            aucs[i_res] = metrics.auc(fpr, tpr)
            times[i_res] = time_sum
            i_res = i_res + 1
            # restart timer
            start = timeit.default_timer()
    return aucs, times

if __name__ == '__main__':
    from utils import data_processing, get_res_idx
    from sklearn.model_selection import train_test_split
    options = dict()
    options['n_pass'] = 20
    options['rec_log'] = .5
    options['rec'] = 50
    options['log_res'] = False
    options['eta'] = 0.1
    options['beta'] = 0.1
    x, y = data_processing('a1a')

    x_tr, x_te, y_tr, y_te = train_test_split(x, y)
    options['res_idx'] = get_res_idx(options['n_pass'] * (len(y_tr) - 1), options)

    aucs, times = auc_fifo_loglink(x_tr, y_tr, x_te, y_te, options)
    print('-------------')
    print('Our Algorithm')
    print('-------------')
    print('AUC Score')
    print(aucs)
    print('-------------')
    print('Time Elapsed')
    print(times)