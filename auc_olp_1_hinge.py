import numpy as np
from sklearn import metrics
import timeit
from utils import get_idx, get_res_idx

def rs_x_t(buf_ids, idx, buf_size, t):
    '''
    Reservoir sampling x at t = s+1
    :param buf_ids:
    :param idx:
    :param buf_size:
    :param n:
    :return:
    '''
    for i in range(buf_size):
        rd = np.random.randint(buf_size + 1)
        if rd == buf_size:
            buf_ids[i] = idx
        else:
            buf_ids[i] = buf_ids[rd]
    return buf_ids

def rs_xx_t(buf_ids, idx, buf_size, t):
    '''
    Reservoir sampling xx at t = s+1
    :param buf_ids:
    :param idx:
    :param buf_size:
    :param n:
    :return:
    '''
    # run through buffer size
    for i in range(buf_size):
        rd = np.random.randint(buf_size + 1)
        if rd == buf_size:
            buf_ids[i] = idx
        else:
            buf_ids[i] = buf_ids[rd]
    return buf_ids

def rs_x(buf_ids, idx, buf_size, n):
    '''
    Reservoir sampling x
    :param buf_ids: the current buffer
    :param idx: sample at t
    :param buf_size: fixed buffer size
    :param n: number of received until t
    :return:
    '''
    buf_ids = np.array(buf_ids)
    mask = [np.random.rand(1)[0] < 1. / n for i in range(buf_size)]
    buf_ids[mask] = idx
    list(buf_ids)
    return buf_ids

def rs_xx(buf_ids, idx, buf_size, n):
    '''
    Reservoir sampling xx
    :param buf_ids: the current buffer
    :param idx: sample at t
    :param buf_size: fixed buffer size
    :param n: number of received until t
    :return:
    '''
    # number of update
    k = np.random.binomial(buf_size, 1. / n)
    if k == 0:
        return buf_ids
    else:
        for i in range(k):
            idx_1 = np.random.randint(buf_size)
            buf_ids[idx_1] = idx
    return buf_ids

def auc_olp_1_hinge(x_tr, y_tr, x_te, y_te, options):
    n_tr,dim = x_tr.shape
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
    # etas = np.ones(n_iter) / (eta * np.sqrt(n_iter)) # each iteration eta
    etas = [eta / np.sqrt(i) for i in range(1, 1 + n_iter)]
    beta = options['beta']
    buf_size = options['buffer']
    buf_ids = [] # initiate buffer
    buf_ids_pos = []  # initiate buffer
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
        eta_t = etas[t]
        if t < buf_size:
            buf_ids.append(ids[t])
        elif t == buf_size:
            buf_ids = rs_xx_t(buf_ids, ids[t], buf_size, t)
        else:
            buf_ids = rs_xx_t(buf_ids, ids[t], buf_size, t)
        x_t_1 = x_tr[buf_ids]
        y_t_1 = y_tr[buf_ids]
        # if y_t == 1:
        #     idx_neg = y_t_1 < 0
        #     n_neg = np.sum(idx_neg)
        #     if n_neg == 0:
        #         xx = np.zeros((1, dim))
        #     else:
        #         xx = x_t - x_t_1[idx_neg]
        #         idx = ((xx @ w_t) <= 1)
        #         w_t = w_t + eta_t * np.sum(xx[idx], axis=0) / buf_size
        # else:
        #     idx_pos = y_t_1 > 0
        #     n_pos = np.sum(idx_pos)
        #     if n_pos == 0:
        #         xx = np.zeros((1, dim))
        #     else:
        #         xx = x_t_1[idx_pos] - x_t
        #         idx = ((xx @ w_t) >= -1)
        #         w_t = w_t - eta_t * np.sum(xx[idx], axis=0) / buf_size
        # wxx = np.dot(xx, w_t)
        # gd = - xx * np.exp(-wxx)[:, None] / (1. + np.exp(-wxx)[:, None])
        # gradient step
        # w_t = w_t - eta_t * np.sum(gd, axis=0) / buf_size
        if y_t == 1:
            n_pos += 1
            if t < buf_size:
                buf_ids_pos.append(ids[t])
            elif t == buf_size:
                buf_ids_pos = rs_xx_t(buf_ids_pos, ids[t], len(buf_ids_pos), t)
            else:
                buf_ids_pos = rs_xx(buf_ids_pos, ids[t], len(buf_ids_pos), t)
            # else:
            #     buf_ids_pos = rs(buf_ids_pos, ids[t], len(buf_ids_pos), t)
            xx = x_t - x_tr[buf_ids_neg]
            idx = ((xx @ w_t) <= 1)
            w_t = w_t + eta_t * np.sum(xx[idx], axis=0) / (1 + len(buf_ids_neg))
        else:
            n_neg += 1
            if t < buf_size:
                buf_ids_neg.append(ids[t])
            elif t == buf_size:
                buf_ids_neg = rs_xx_t(buf_ids_neg, ids[t], len(buf_ids_neg), t)
            else:
                buf_ids_neg = rs_xx(buf_ids_neg, ids[t], len(buf_ids_neg), t)
            # else:
            #     buf_ids_neg = rs(buf_ids_neg, ids[t], len(buf_ids_neg), t)
            xx = x_t - x_tr[buf_ids_pos]
            idx = ((xx @ w_t) >= -1)
            w_t = w_t - eta_t * np.sum(xx[idx], axis=0) / (1 + len(buf_ids_pos))
        # gradient step
        # w_t = w_t - eta_t * gd

        # projection
        norm = np.linalg.norm(w_t)
        if norm > beta:
            w_t = w_t * beta / norm
        # naive average
        w_sum = w_sum + eta_t * w_t
        eta_sum = eta_sum + eta_t
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

def rs_x_t_(buf, buf_y, x, y, buf_size, t):
    '''
    Reservoir sampling x at t = s+1
    :param buf_ids:
    :param idx:
    :param buf_size:
    :param n:
    :return:
    '''
    for i in range(buf_size):
        rd = np.random.randint(buf_size + 1)
        if rd == buf_size:
            buf[i] = x
            buf_y[i] = y
        else:
            buf[i] = buf[rd]
            buf_y[i] = buf_y[rd]
    return buf, buf_y

def rs_x_(buf, buf_y, x, y, buf_size, n):
    '''
    Reservoir sampling x
    :param buf_ids: the current buffer
    :param idx: sample at t
    :param buf_size: fixed buffer size
    :param n: number of received until t
    :return:
    '''
    mask = [np.random.rand(1)[0] < 1. / n for i in range(buf_size)]
    buf[mask] = x
    buf_y[mask] = y
    return buf, buf_y

def rs_xx_t_(buf, buf_y, x, y, buf_size, t):
    '''
    Reservoir sampling xx at t = s+1
    :param buf_ids:
    :param idx:
    :param buf_size:
    :param n:
    :return:
    '''
    # run through buffer size
    for i in range(buf_size):
        rd = np.random.randint(buf_size + 1)
        if rd == buf_size:
            buf[i] = x
            buf_y[i] = y
        else:
            buf[i] = buf[rd]
            buf_y[i] = buf_y[rd]
    return buf, buf_y

def rs_xx_(buf, buf_y, x, y, buf_size, n):
    '''
    Reservoir sampling xx
    :param buf_ids: the current buffer
    :param idx: sample at t
    :param buf_size: fixed buffer size
    :param n: number of received until t
    :return:
    '''
    # number of update
    k = np.random.binomial(buf_size, 1. / n)
    if k == 0:
        return buf
    else:
        for i in range(k):
            idx_1 = np.random.randint(buf_size)
            buf[idx_1] = x
            buf_y[idx_1] = y
        return buf, buf_y

def auc_olp_hinge_(x_tr, y_tr, x_te, y_te, options):
    n_tr,dim = x_tr.shape
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
    # etas = np.ones(n_iter) / (eta * np.sqrt(n_iter)) # each iteration eta
    etas = [eta / np.sqrt(i) for i in range(1, 1 + n_iter)]
    beta = options['beta']
    buf_size = options['buffer']
    buf = np.zeros((buf_size, dim)) # initiate buffer
    buf_y = np.zeros(buf_size)
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
        eta_t = etas[t]
        if len(buf) < buf_size:
            buf[t] = x_t
            buf_y[t] = y_t
        elif len(buf) == buf_size:
            buf, buf_y = rs_xx_t_(buf, buf_y, x_t, y_t, buf_size, t)
        else:
            buf, buf_y = rs_xx_(buf, buf_y, x_t, y_t, buf_size, t)
        if y_t == 1:
            xx = x_t - buf[buf_y < 0]
            idx = ((xx @ w_t) <= 1)
            # gradient step
            w_t = w_t + eta_t * np.sum(xx[idx], axis=0) / buf_size
        else:
            xx = buf[buf_y > 0] - x_t
            idx = ((xx @ w_t) >= -1)
            # gradient step
            w_t = w_t - eta_t * np.sum(xx[idx], axis=0) / buf_size

        # if y_t == 1:
        #     n_pos += 1
        #     if n_pos + n_neg < buf_size:
        #         buf_ids_pos.append(ids[t])
        #     elif n_neg + n_neg == buf_size:
        #         buf_ids_pos = rs_x_t(buf_ids_pos, ids[t], len(buf_ids_pos), t)
        #     else:
        #         buf_ids_pos = rs_x(buf_ids_pos, ids[t], len(buf_ids_pos), t)
        #     # else:
        #     #     buf_ids_pos = rs(buf_ids_pos, ids[t], len(buf_ids_pos), t)
        #     xx = x_t - x_tr[buf_ids_neg]
        #     idx = ((xx @ w_t) <= 1)
        #     w_t = w_t + eta_t * np.sum(xx[idx], axis=0)
        # else:
        #     n_neg += 1
        #     if n_neg + n_neg < buf_size:
        #         buf_ids_neg.append(ids[t])
        #     elif n_neg + n_neg == buf_size:
        #         buf_ids_neg = rs_x_t(buf_ids_neg, ids[t], len(buf_ids_neg), t)
        #     else:
        #         buf_ids_neg = rs_x(buf_ids_neg, ids[t], len(buf_ids_neg), t)
        #     # else:
        #     #     buf_ids_neg = rs(buf_ids_neg, ids[t], len(buf_ids_neg), t)
        #     xx = x_t - x_tr[buf_ids_pos]
        #     idx = ((xx @ w_t) >= -1)
        #     w_t = w_t - eta_t * np.sum(xx[idx], axis=0)
        # # gradient step
        # w_t = w_t - eta_t * gd

        # projection
        norm = np.linalg.norm(w_t)
        if norm > beta:
            w_t = w_t * beta / norm
        # naive average
        w_sum = w_sum + eta_t * w_t
        eta_sum = eta_sum + eta_t
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
    options['rec_log'] = .1
    options['rec'] = 50
    options['log_res'] = False
    options['eta'] = .01
    options['beta'] = 0.1
    options['buffer'] = 1

    x, y = data_processing('diabetes')
    x_tr, x_te, y_tr, y_te = train_test_split(x, y)
    aucs, times = auc_olp_1_hinge(x_tr, y_tr, x_te, y_te, options)
    print('-------------')
    print('All Past Algorithm')
    print('-------------')
    print('AUC Score')
    print(aucs)
    print('-------------')
    print('Time Elapsed')
    print(times)