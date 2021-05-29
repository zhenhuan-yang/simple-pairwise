import numpy as np
from sklearn import metrics
import timeit
from utils import get_idx, get_stage_idx

def sigma_calibrator(epsilon, delta, eta):
    sigma = 4. * eta * np.sqrt(np.log(4. / delta)**3) / epsilon
    return sigma

def auc_fifo_stage_logistic(stage, w_init, x_tr, y_tr, options):
    n_tr = len(y_tr)
    dim = options['dim']
    n_pass = options['n_pass']
    # if stage == 0:
    #     n_pass = 50
    # else:
    #     n_pass = options['n_pass']
    start = timeit.default_timer()
    ids = get_idx(n_tr, n_pass) # indices in this partition
    n_iter = len(ids)
    stop = timeit.default_timer()
    time_avg = (stop - start) / n_iter
    w_t = w_init + 0. # warm start

    w_sum = np.zeros(dim)
    w_sum_old = np.zeros(dim)
    eta_sum = 0.
    eta_sum_old = 0.
    eta_t = options['eta_stage'] # eta for this partition
    beta = options['beta']
    t = 0
    time_sum = 0.
    while t < n_iter:
        start = timeit.default_timer()
        # current example
        y_t = y_tr[ids[t]]
        # past example
        y_t_1 = y_tr[ids[t - 1]]
        # when ys are different
        if y_t * y_t_1 < 0:
            x_t = x_tr[ids[t]]
            x_t_1 = x_tr[ids[t - 1]]  # start with -1, it suppose to be random
            # make sure positive is in front
            yxx = y_t * (x_t - x_t_1)
            wyxx = np.inner(w_t, yxx)
            # to avoid overflow
            # if np.abs(wyxx) > 100.:
            #     print('warning: overflow')
            #     wyxx = np.sign(wyxx) * 100.
            # logistic loss gradient
            gd = - yxx * np.exp(-wyxx) / (1. + np.exp(-wyxx))
            # gradient step
            w_t = w_t - eta_t * gd
            # w_t = w_t_1 - eta_t * gd
            # projection
            if options['proj_flag']:
                # projection
                norm = np.linalg.norm(w_t)
                if norm > beta:
                    w_t = w_t * beta / norm
        w_sum = w_sum + w_t
        eta_sum += 1
        stop = timeit.default_timer()
        time_sum += stop - start
        # update
        t = t + 1
    w_avg = (w_sum - w_sum_old) / (eta_sum - eta_sum_old)
    time_sum += time_avg * (eta_sum - eta_sum_old)
    return w_avg, time_sum

def auc_dp_fifo_logistic(x_tr, y_tr, x_te, y_te, options):
    # get
    n_tr = options['n_tr']
    dim = options['dim']
    stages = get_stage_idx(n_tr) # stages
    w_t = np.zeros(dim)
    w_priv = np.zeros(dim)
    eta = options['eta'] # initial eta
    etas = eta / (4**np.arange(1, len(stages) + 1)) # stagewise
    # etas = [1. / (.001 + eta * np.sqrt(i)) for i in range(n_iter)] # each iteration eta
    aucs = np.zeros(len(stages))
    times = np.zeros(len(stages))
    stage = 0
    while stage < len(stages):
        # return the data for this stage
        x_tr_stage = x_tr[stages[stage]]
        y_tr_stage = y_tr[stages[stage]]
        # eta for this stage
        options['eta_stage'] = etas[stage]
        # warm start
        w_t, time_t = auc_fifo_stage_logistic(stage, w_priv, x_tr_stage, y_tr_stage, options)
        # start timer
        start = timeit.default_timer()
        # calibrate noise
        sigma = sigma_calibrator(options['epsilon'], options['delta'], options['eta_stage'])
        b_t = np.random.normal(0., sigma, dim)
        # print('w_t: %.4f b_t: %.4f' %(np.linalg.norm(w_t), np.linalg.norm(b_t)))
        w_priv = w_t + b_t
        stop = timeit.default_timer()
        time_t += stop - start
        # calculate auc
        pred = (x_te.dot(w_priv.T)).ravel()
        if not np.all(np.isfinite(pred)):
            break
        fpr, tpr, thresholds = metrics.roc_curve(y_te, pred.T, pos_label=1)
        aucs[stage] = metrics.auc(fpr, tpr)
        times[stage] = times[stage - 1] + time_t
        # update
        stage = stage + 1
    return aucs, times

if __name__ == '__main__':
    from utils import data_processing, get_res_idx, get_stage_res_idx

    from sklearn.model_selection import train_test_split, RepeatedKFold
    options = dict()
    options['rec_log'] = .5
    options['rec'] = 50
    options['log_res'] = False
    options['eta'] = .01
    options['beta'] = 1.
    options['epsilon'] = 1.

    options['n_tr'] = 256
    options['n_repeat'] = 5
    options['delta'] = 1. / options['n_tr']
    options['n_pass'] = np.log2(options['n_tr']**2).astype(int) + 1
    options['n_pass'] = 100
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
        tr_idx = tr_idx[:options['n_tr']]
        x_tr, x_te = x[tr_idx], x[te_idx]
        y_tr, y_te = y[tr_idx], y[te_idx]
        auc_t, time_t = auc_dp_fifo_logistic(x_tr, y_tr, x_te, y_te, options)
        time_rec[i] = time_t
        auc_rec[i] = auc_t
    print('----------------------')
    print(np.mean(auc_rec, axis=0))
    print(np.std(auc_rec, axis=0))