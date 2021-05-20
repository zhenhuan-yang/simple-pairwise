import numpy as np
from sklearn import metrics
import timeit
from utils import get_pair_idx

def auc_pair_hinge(x_tr, y_tr, x_te, y_te, options):
    # get
    n_tr, dim = x_tr.shape
    n_pass = options['n_pass']
    # need to calculate every time in order to align record and training
    n_iter = n_tr * n_pass
    ids = get_pair_idx(n_tr, n_pass)
    res_idx = options['res_idx']
    w_t = np.zeros(dim)
    # w_t_1 = np.zeros(dim)
    # w_t_2 = np.zeros(dim)
    w_sum = np.zeros(dim)
    w_sum_old = np.zeros(dim)
    eta_sum = 0.
    eta_sum_old = 0.
    eta = options['eta'] # initial eta
    # etas = [1. / (.001 + eta * np.sqrt(i)) for i in range(n_iter)] # each iteration eta
    etas = eta * np.ones(n_iter) / np.sqrt(n_iter)
    beta = options['beta']
    aucs = np.zeros(len(res_idx))
    times = np.zeros(len(res_idx))
    t = 0
    i_res = 0
    time_sum = 0.
    # initiate timer
    start = timeit.default_timer()
    while t < len(ids):
        # sample a pair of samples without replacement
        # idx = np.random.choice(n_tr, size=2, replace=False)
        # current example
        x_t = x_tr[ids[t][0]]
        y_t = y_tr[ids[t][0]]
        # the other example
        x_t_1 = x_tr[ids[t][1]]
        y_t_1 = y_tr[ids[t][1]]
        eta_t = etas[t]
        # when ys are different
        if y_t * y_t_1 < 0:
            # make sure positive is in front
            yxx = y_t * (x_t - x_t_1)
            wyxx = np.inner(w_t, yxx)
            # hinge loss gradient
            if 1 - wyxx > 0.:
                gd = - yxx
            else:
                gd = 0.
            # gradient step
            w_t = w_t - eta_t * gd
            # projection
            norm = np.linalg.norm(w_t)
            if norm > beta:
                w_t = w_t * beta / norm
        # naive average
        w_sum = w_sum + eta_t * w_t
        eta_sum = eta_sum + eta_t
        # update
        t = t + 1
        # save results
        if i_res < len(res_idx) and res_idx[i_res] == t:
            # stop timer
            stop = timeit.default_timer()
            time_sum += stop - start
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
    options['n_pass'] = 3
    options['rec_log'] = .5
    options['rec'] = 50
    options['log_res'] = False
    options['eta'] = 0.1
    options['beta'] = 0.1

    x, y = data_processing('diabetes')
    x_tr, x_te, y_tr, y_te = train_test_split(x, y)
    options['res_idx'] = get_res_idx(options['n_pass'] * (len(y_tr) - 1), options)
    aucs, times = auc_pair_hinge(x_tr, y_tr, x_te, y_te, options)
    print('-------------')
    print('Our Algorithm')
    print('-------------')
    print('AUC Score')
    print(aucs)
    print('-------------')
    print('Time Elapsed')
    print(times)