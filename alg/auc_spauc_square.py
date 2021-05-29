import numpy as np
from sklearn import metrics
import timeit
from utils import get_idx

def auc_spauc_square(x_tr, y_tr, x_te, y_te, options):
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
    idx_pos = (y_tr == 1)
    idx_neg = (y_tr == -1)
    p = np.sum(idx_pos) / n_tr
    ax_pos = np.mean(x_tr[idx_pos, :], axis=0)
    ax_neg = np.mean(x_tr[idx_neg, :], axis=0)
    diff = ax_neg - ax_pos
    eta_sum = 0.
    eta_sum_old = 0.
    eta = options['eta'] # initial eta
    # if options['eta_geo'] == 'const':
    #     etas = eta * np.ones(n_iter) / np.sqrt(n_iter)
    # elif options['eta_geo'] == 'sqrt':
    #     etas = eta / np.sqrt(np.arange(1, n_iter + 1))
    # elif options['eta_geo'] == 'fast':
    #     eta = 1. / eta
    #     etas = 2 / (eta * np.arange(1, n_iter + 1) + 1.)
    # else:
    #     print('Wrong step size geometry!')
    #     etas = eta / np.sqrt(np.arange(1, n_iter + 1))
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
        x_t = x_tr[ids[t]]
        y_t = y_tr[ids[t]]
        eta_t = etas[t]
        t += 1
        if y_t > 0:
            minus = x_t - ax_pos
            gd = (1 - p) * np.inner(w_t, minus) * minus
        else:
            minus = x_t - ax_neg
            gd = p * np.inner(w_t, minus) * minus
        gd = gd + p * (1 - p) * (np.inner(diff, w_t) + 1) * diff
        w_t = w_t - eta_t * gd
        if options['proj_flag']:
            # projection
            norm = np.linalg.norm(w_t)
            if norm > beta:
                w_t = w_t * beta / norm
        # naive average
        w_sum = w_sum + w_t
        eta_sum = eta_sum + 1
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
            eta_sum_old = eta_sum
            w_sum_old = w_sum
            # restart timer
            start = timeit.default_timer()
    return aucs, times

def auc_spauc_square_(x_tr, y_tr, x_te, y_te, options):
    # get
    n_tr = options['n_tr']
    dim = options['dim']
    n_pass = options['n_pass']
    # start = timeit.default_timer()
    # ids = get_pair_idx(n_tr, n_pass)
    # stop = timeit.default_timer()
    # time_avg = (stop - start) / n_iter
    ids = options['ids']
    # need to calculate every time in order to align record and training
    n_iter = len(ids)
    time_avg = options['time_avg']
    res_idx = options['res_idx']
    w_t = np.zeros(dim)
    # w_t_1 = np.zeros(dim)
    # w_t_2 = np.zeros(dim)
    w_sum = np.zeros(dim)
    w_sum_old = np.zeros(dim)
    sx_pos = np.zeros(dim)  # summation of positive instances
    sx_neg = np.zeros(dim)  # summation of negative instances
    ax_pos = sx_pos  # average of positive instances
    ax_neg = sx_neg  # average of positive instances
    eta_sum = 0.
    eta_sum_old = 0.
    eta = options['eta'] # initial eta
    # if options['eta_geo'] == 'const':
    #     etas = eta * np.ones(n_iter) / np.sqrt(n_iter)
    # elif options['eta_geo'] == 'sqrt':
    #     etas = eta / np.sqrt(np.arange(1, n_iter + 1))
    # elif options['eta_geo'] == 'fast':
    #     eta = 1. / eta
    #     etas = 2 / (eta * np.arange(1, n_iter + 1) + 1.)
    # else:
    #     print('Wrong step size geometry!')
    #     etas = eta / np.sqrt(np.arange(1, n_iter + 1))
    etas = options['etas']
    beta = options['beta']
    aucs = np.zeros(len(res_idx))
    times = np.zeros(len(res_idx))
    t = 0
    sp = 0
    i_res = 0
    time_sum = 0.
    # initiate timer
    start = timeit.default_timer()
    while t < n_iter:
        # current example
        x_t = x_tr[ids[t]]
        y_t = y_tr[ids[t]]
        eta_t = etas[t]
        t += 1
        if y_t > 0:
            sp += 1
            p = sp / t
            sx_pos = sx_pos + x_t
            ax_pos = sx_pos / sp
            diff = ax_neg - ax_pos
            minus = x_t - ax_pos
            gd = (1 - p) * np.inner(w_t, minus) * minus
        else:
            p = sp / t
            sx_neg = sx_neg + x_t
            ax_neg = sx_neg / (t - sp)
            diff = ax_neg - ax_pos
            minus = x_t - ax_neg
            gd = p * np.inner(w_t, minus) * minus
        gd = gd + p * (1 - p) * (np.inner(diff, w_t) + 1) * diff
        w_t = w_t - eta_t * gd
        if options['proj_flag']:
            # projection
            norm = np.linalg.norm(w_t)
            if norm > beta:
                w_t = w_t * beta / norm
        # naive average
        w_sum = w_sum + w_t
        eta_sum = eta_sum + 1
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

def auc_spauc_square__(x_tr, y_tr, x_te, y_te, options):
    # get
    n_tr, dim = x_tr.shape
    n_pass = options['n_pass']
    # need to calculate every time in order to align record and training
    n_iter = n_tr * n_pass
    ids = get_idx(n_tr, n_pass)
    res_idx = options['res_idx']
    w_t = np.zeros(dim)
    # w_t_1 = np.zeros(dim)
    # w_t_2 = np.zeros(dim)
    w_sum = np.zeros(dim)
    w_sum_old = np.zeros(dim)
    sp = 0
    sx_pos = np.zeros(dim)  # summation of positive instances
    sx_neg = np.zeros(dim)  # summation of negative instances
    ax_pos = sx_pos  # average of positive instances
    ax_neg = sx_neg  # average of positive instances
    diff = np.zeros(dim)
    eta_sum = 0.
    eta_sum_old = 0.
    eta = options['eta'] # initial eta
    if options['eta_geo'] == 'const':
        etas = eta * np.ones(n_iter) / np.sqrt(n_iter)
    elif options['eta_geo'] == 'sqrt':
        etas = eta / np.sqrt(np.arange(1, n_iter + 1))
    elif options['eta_geo'] == 'fast':
        eta = 1. / eta
        etas = 2 / (eta * np.arange(1, n_iter + 1) + 1.)
    else:
        print('Wrong step size geometry!')
        etas = eta / np.sqrt(np.arange(1, n_iter + 1))
    beta = options['beta']
    aucs = np.zeros(len(res_idx))
    times = np.zeros(len(res_idx))
    t = 0
    i_res = 0
    time_sum = 0.
    # initiate timer
    if 2 * n_tr < n_iter:
        t_stage = 2 * n_tr
    else:
        t_stage = n_tr
    start = timeit.default_timer()
    while t < t_stage:
        # current example
        x_t = x_tr[ids[t]]
        y_t = y_tr[ids[t]]
        eta_t = etas[t]
        t += 1
        if y_t > 0:
            sp += 1
            p = sp / t
            sx_pos = sx_pos + x_t
            ax_pos = sx_pos / sp
            diff = ax_neg - ax_pos
            minus = x_t - ax_pos
            gd = (1 - p) * np.inner(w_t, minus) * minus
        else:
            p = sp / t
            sx_neg = sx_neg + x_t
            ax_neg = sx_neg / (t - sp)
            diff = ax_neg - ax_pos
            minus = x_t - ax_neg
            gd = p * np.inner(w_t, minus) * minus
        gd = gd + p * (1 - p) * (np.inner(diff, w_t) + 1) * diff
        w_t = w_t - eta_t * gd
        # naive average
        w_sum = w_sum + w_t
        eta_sum = eta_sum + 1
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

    diff_coef = 0
    diff_coef_s = 0
    diff_coef_s_old = 0
    diff_square = np.inner(diff, diff)
    #    print('diff_square=%.4f' % diff_square)
    # the most time-consumping is the calculation with vector output, the idea is at each iteration nor record the vector but its coefficient at the base vector
    # diff, using the idea of kernel learning. This can speed up the training process

    while t < len(ids):
        x_t = x_tr[ids[t]]
        y_t = y_tr[ids[t]]
        eta = etas[t]
        t = t + 1
        if y_t == 1:
            minus = x_t - ax_pos
            tmp = np.inner(w_t, minus) + diff_coef * np.inner(diff, minus)
            gd = (1 - p) * tmp * minus
        else:
            minus = x_t - ax_neg
            tmp = np.inner(w_t, minus) + diff_coef * np.inner(diff, minus)
            gd = p * tmp * minus
        tmp = np.inner(diff, w_t) + diff_coef * diff_square + 1
        w_t = w_t - eta * gd
        diff_coef = diff_coef - eta * p * (1 - p) * tmp
        diff_coef_s = diff_coef_s + diff_coef
        w_sum = w_sum + w_t  # * eta
        eta_sum += 1  # eta
        if i_res < len(res_idx) and res_idx[i_res] == t:
            stop = timeit.default_timer()
            time_sum += stop - start
            times[i_res] = time_sum
            #            if np.any(np.isnan(w)):# or not np.any(y_te == 1):
            ##                print('NAN encountered')
            #                break
            tmp = (diff_coef_s - diff_coef_s_old) * diff
            w_ave = (w_sum - w_sum_old + tmp) / (eta_sum - eta_sum_old)
            pred = (x_te.dot(w_ave.T)).ravel()
            if not np.all(np.isfinite(pred)):
                aucs[i_res:] = aucs[i_res - 1]
                times[i_res:] = time_sum
                break
            fpr, tpr, thresholds = metrics.roc_curve(y_te, pred.T, pos_label=1)
            aucs[i_res] = metrics.auc(fpr, tpr)
            w_sum_old = w_sum
            eta_sum_old = eta_sum
            diff_coef_s_old = diff_coef_s
            i_res = i_res + 1
            start = timeit.default_timer()
    return aucs, times


if __name__ == '__main__':
    from utils import data_processing, get_res_idx

    from sklearn.model_selection import train_test_split
    options = dict()
    options['n_pass'] = 3
    options['n_tr'] = 500
    options['rec_log'] = .5
    options['rec'] = 50
    options['log_res'] = True
    options['eta'] = 10.
    options['beta'] = 10.
    options['eta_geo'] = 'const'
    options['proj_flag'] = True
    # start = timeit.default_timer()
    # options['ids'] = get_idx(options['n_tr'], options['n_pass'])
    # stop = timeit.default_timer()
    # options['time_avg'] = (stop - start) / (options['n_pass'] * options['n_tr'])
    options['etas'] = options['eta'] / np.sqrt(np.arange(1, options['n_pass'] * options['n_tr'] + 1))
    x, y = data_processing('diabetes')
    _, options['dim'] = x.shape
    auc_sum = np.zeros(5)
    time_sum = np.zeros(5)
    for i in range(5):
        x_tr, x_te, y_tr, y_te = train_test_split(x, y, train_size=options['n_tr'])
        options['res_idx'] = get_res_idx(options['n_pass'] * (options['n_tr'] - 1), options)
        aucs, times = auc_spauc_square(x_tr, y_tr, x_te, y_te, options)
        auc_sum[i] = aucs[-1]
        time_sum[i] = times[-1]

    print('AUC Score: %f \pm %f Time Elapsed: %f \pm %f' % (
    np.mean(auc_sum), np.std(auc_sum), np.mean(time_sum), np.std(time_sum)))