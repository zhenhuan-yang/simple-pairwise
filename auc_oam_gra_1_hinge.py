import numpy as np
from sklearn import metrics
import timeit
from utils import get_idx, get_res_idx

def rs_(buf_ids, idx, buf_size, n):
    '''
    Reservoir sampling
    :param buf_ids: the current buffer
    :param idx: sample at t
    :param buf_size: fixed buffer size
    :param n: number of received until t
    :return:
    '''
    rd = np.random.rand(1)
    if rd[0] < buf_size / n:
        idx_1 = np.random.randint(buf_size)
        buf_ids[idx_1] = idx
    return buf_ids

def auc_oam_gra_hinge_(x_tr, y_tr, x_te, y_te, options):
    n_tr, dim = x_tr.shape
    n_pass = options['n_pass']
    n_iter = n_tr * n_pass
    ids = get_idx(n_tr, n_pass)
    res_idx = get_res_idx(n_iter, options)
    w_t = np.zeros(dim)
    w_sum = np.zeros(dim)
    w_sum_old = np.zeros(dim)
    eta_sum = 0.
    eta_sum_old = 0.
    eta = options['eta'] # initial eta
    # etas = eta * np.ones(n_iter) / np.sqrt(n_iter) # each iteration eta
    # etas = [1. / (.001 + eta * np.sqrt(i)) for i in range(n_iter)]
    beta = options['beta']
    buf_size = options['buffer']
    buf_ids_pos = [] # initiate buffer
    buf_ids_neg = []
    n_pos = 0
    n_neg = 0
    aucs = np.zeros(len(res_idx))
    times = np.zeros(len(res_idx))
    t = 0
    i_res = 0
    time_sum = 0.
    # initiate timer
    start = timeit.default_timer()
    while t < len(ids):
        # current example
        x_t = x_tr[ids[t]]
        y_t = y_tr[ids[t]]
        if y_t == 1:
            n_pos += 1
            C_t = eta * max(1, n_neg / buf_size)
            if n_pos <= buf_size:
                buf_ids_pos.append(ids[t])
            else:
                buf_ids_pos = rs_(buf_ids_pos, ids[t], buf_size, n_pos)
            xx = x_t - x_tr[buf_ids_neg]
            idx = ((xx @ w_t) <= 1)
            w_t = w_t + C_t * np.sum(xx[idx], axis=0)
        else:
            n_neg += 1
            C_t = eta * max(1, n_pos / buf_size)
            if n_neg <= buf_size:
                buf_ids_neg.append(ids[t])
            else:
                buf_ids_neg = rs_(buf_ids_neg, ids[t], buf_size, n_neg)
            xx = x_t - x_tr[buf_ids_pos]
            idx = ((xx @ w_t) >= -1)
            w_t = w_t - C_t * np.sum(xx[idx], axis=0)

        # naive average
        w_sum = w_sum + eta * w_t
        eta_sum = eta_sum + eta
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

def rs(buf, x, buf_size, n):
    '''
    Reservoir sampling
    :param buf_ids: the current buffer
    :param idx: sample at t
    :param buf_size: fixed buffer size
    :param n: number of received until t
    :return:
    '''
    rd = np.random.rand(1)
    if rd[0] < buf_size / n:
        idx_1 = np.random.randint(buf_size)
        buf[idx_1] = x
    return buf

def auc_oam_gra_1_hinge(x_tr, y_tr, x_te, y_te, options):
    n_tr, dim = x_tr.shape
    n_pass = options['n_pass']
    n_iter = n_tr * n_pass
    ids = get_idx(n_tr, n_pass)
    res_idx = get_res_idx(n_iter, options)
    w_t = np.zeros(dim)
    w_sum = np.zeros(dim)
    w_sum_old = np.zeros(dim)
    eta_sum = 0.
    eta_sum_old = 0.
    eta = options['eta'] # initial eta
    # etas = eta * np.ones(n_iter) / np.sqrt(n_iter) # each iteration eta
    # etas = [1. / (.001 + eta * np.sqrt(i)) for i in range(n_iter)]
    beta = options['beta']
    buf_size = options['buffer']
    buf_pos = np.zeros((buf_size, dim)) # initiate buffer
    buf_neg = np.zeros((buf_size, dim))
    n_pos = 0
    n_neg = 0
    aucs = np.zeros(len(res_idx))
    times = np.zeros(len(res_idx))
    t = 0
    i_res = 0
    time_sum = 0.
    # initiate timer
    start = timeit.default_timer()
    while t < len(ids):
        # current example
        x_t = x_tr[ids[t]]
        y_t = y_tr[ids[t]]
        if y_t == 1:
            n_pos += 1
            C_t = eta * max(1, n_neg / buf_size)
            if n_pos <= buf_size:
                buf_pos[n_pos - 1] = x_t
            else:
                buf_pos = rs(buf_pos, x_t, buf_size, n_pos)
            xx = x_t - buf_neg
            idx = ((xx @ w_t) <= 1)
            w_t = w_t + C_t * np.sum(xx[idx], axis=0)
        else:
            n_neg += 1
            C_t = eta * max(1, n_pos / buf_size)
            if n_neg <= buf_size:
                buf_neg[n_neg - 1] = x_t
            else:
                buf_neg = rs(buf_neg, x_t, buf_size, n_neg)
            xx = x_t - buf_pos
            idx = ((xx @ w_t) >= -1)
            w_t = w_t - C_t * np.sum(xx[idx], axis=0)

        # naive average
        w_sum = w_sum + w_t
        eta_sum = eta_sum + 1
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
    from utils import data_processing

    from sklearn.model_selection import train_test_split
    options = dict()
    options['n_pass'] = 1
    options['rec_log'] = .5
    options['rec'] = 50
    options['log_res'] = False
    options['eta'] = 0.1
    options['beta'] = 0.1
    options['buffer'] = 1

    x, y = data_processing('diabetes')
    x_tr, x_te, y_tr, y_te = train_test_split(x, y)
    aucs, times = auc_oam_gra_1_hinge(x_tr, y_tr, x_te, y_te, options)
    print('-------------')
    print('All Past Algorithm')
    print('-------------')
    print('AUC Score')
    print(aucs)
    print('-------------')
    print('Time Elapsed')
    print(times)