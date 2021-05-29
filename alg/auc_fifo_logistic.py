import numpy as np
from sklearn import metrics
import timeit
from utils import get_idx

def auc_fifo_logistic(x_tr, y_tr, x_te, y_te, options):
    # get
    n_tr = options['n_tr']
    dim = options['dim']
    n_pass = options['n_pass']
    start = timeit.default_timer()
    ids = get_idx(n_tr, n_pass)
    stop = timeit.default_timer()
    n_iter = len(ids)
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
    # if options['eta_geo'] == 'const':
    #     etas = eta * np.ones(n_iter) / np.sqrt(n_iter)
    # elif options['eta_geo'] == 'sqrt':
    #     etas = eta / np.sqrt(np.arange(1, n_iter + 1))
    # elif options['eta_geo'] == 'fast':
    #     eta = 1. / eta
    #     etas = 2 / (eta * np.arange(1, n_iter + 1) + 1.)
    # else:
    #     print('Wrong step size geometry!')
    #     etas = eta * np.ones(n_iter) / np.sqrt(n_iter)
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
            # logistic loss gradient
            gd = - yxx * np.exp(-wyxx) / (1. + np.exp(-wyxx))
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
            w_sum_old = w_sum
            eta_sum_old - eta_sum
            # restart timer
            start = timeit.default_timer()
    return aucs, times

if __name__ == '__main__':
    from utils import data_processing, get_res_idx

    from sklearn.model_selection import train_test_split
    options = dict()
    options['n_pass'] = 1
    options['rec_log'] = .5
    options['rec'] = 50
    options['log_res'] = False
    options['eta'] = 0.01
    options['beta'] = 0.1
    options['eta_geo'] = 'const'
    options['proj_flag'] = True

    x, y = data_processing('diabetes')
    n, options['dim'] = x.shape
    options['n_tr'] = 256
    options['etas'] = options['eta'] / np.sqrt(np.arange(1, options['n_pass'] * options['n_tr'] + 1))

    auc_sum = np.zeros(5)
    time_sum = np.zeros(5)
    for i in range(5):
        x_tr, x_te, y_tr, y_te = train_test_split(x, y, train_size=options['n_tr'])
        options['res_idx'] = get_res_idx(options['n_pass'] * (len(y_tr) - 1), options)

        aucs, times = auc_fifo_logistic(x_tr, y_tr, x_te, y_te, options)
        auc_sum[i] = aucs[-1]
        time_sum[i] = times[-1]
    print('AUC Score: %f \pm %f Time Elapsed: %f \pm %f' % (np.mean(auc_sum), np.std(auc_sum), np.mean(time_sum), np.std(time_sum)))

