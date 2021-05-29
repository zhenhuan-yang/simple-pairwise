import numpy as np
from sklearn import metrics
from utils import get_stage_idx
import timeit

def sigma_calibrator(epsilon, delta, eta):
    sigma = 4. * eta * np.sqrt(np.log(1. / delta)) / epsilon
    return sigma

def auc_egd_stage_logistic(stage, w_init, x_tr, y_tr, options):
    n_tr = len(y_tr)
    dim = options['dim']
    n_pass = options['n_pass']
    w_t = w_init + 0. # warm start
    w_sum = np.zeros(dim)
    w_sum_old = np.zeros(dim)
    eta_sum = 0
    eta_sum_old = 0.
    eta_t = options['eta_stage'] # eta for this partition
    beta = options['beta']
    # get positive and negative indices
    idx_pos = y_tr > 0.
    idx_neg = y_tr < 0.
    n_pos = int(np.sum(idx_pos) / 2)
    n_neg = int(np.sum(idx_neg) / 2)
    # print('n_pos: %d n_neg: %d' % (n_pos, n_neg))
    start = timeit.default_timer()
    # when this partition has empty class then return
    if n_pos == 0 or n_neg == 0:
        stop = timeit.default_timer()
        time_sum = stop - start
        return w_t, time_sum
    # batch_size = min(int(options['batch_size'] / 2), n_pos, n_neg)
    # n_iter = int(n_tr * n_pass / batch_size)
    # print('batch_size: %d n_pos: %d n_neg: %d' %(batch_size, n_pos, n_neg))
    x_pos = x_tr[idx_pos]
    x_neg = x_tr[idx_neg]
    t = 0
    time_sum = 0.
    while t < n_pass:
        # pick same batch size from positive and negative sample to ease computation
        # i_pos = np.random.choice(n_pos, batch_size, replace=False)
        # i_neg = np.random.choice(n_neg, batch_size, replace=False)
        # xx = x_pos[i_pos] - x_neg[i_neg]
        # gds = np.zeros(dim)
        # batch repeat is to boost combination between positive and negative sample
        # for i in range(options['batch_repeat']):
        #     i_pos = np.random.permutation(i_pos) # permute the positive and negative sample
        #     xx = x_pos[i_pos] - x_neg[i_neg]
        #
        #     # gradient on this repeat
        #     gd = - xx * np.exp(-wxx)[:, None] / (1. + np.exp(-wxx)[:, None])
        #     gds += np.sum(gd, axis=0)
        # start timer
        start = timeit.default_timer()
        # total gradient at this iteration
        gds = np.zeros(dim)
        stop = timeit.default_timer()
        time_sum += stop - start
        # to match positive with negative
        for i in range(n_pos):
            start = timeit.default_timer()
            xx = x_pos[i] - x_neg
            wxx = np.dot(xx, w_t)
            # avoid overflow
            # mask = np.abs(wxx) > 100.
            # sign = np.sign(wxx)
            # wxx[mask] = sign[mask] * 100.
            gd = - xx * np.exp(-wxx)[:, None] / (1. + np.exp(-wxx)[:, None])
            # update total gradient
            gds += np.sum(gd, axis=0)
            stop = timeit.default_timer()
            time_sum += stop - start
        start = timeit.default_timer()
        # gradient descent
        w_t = w_t - eta_t * gds / (n_pos * n_neg)
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
    # print('n_iter: %d w_sum: %f eta_sum: %f' % (n_iter, np.linalg.norm(w_sum), eta_sum))
    w_avg = w_sum / eta_sum
    return w_avg, time_sum

def auc_dp_egd_logistic(x_tr, y_tr, x_te, y_te, options):
    # get
    n_tr = options['n_tr']
    dim = options['dim']
    stages = get_stage_idx(n_tr) # stages
    w_t = np.zeros(dim)
    w_priv = np.zeros(dim)
    eta = options['eta'] # initial eta
    etas = eta / (4**np.arange(1, len(stages)+1)) # stagewise
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
        w_t, time_t = auc_egd_stage_logistic(stage, w_priv, x_tr_stage, y_tr_stage, options)
        # start timer
        start = timeit.default_timer()
        # calibrate noise
        sigma = sigma_calibrator(options['epsilon'], options['delta'], options['eta_stage'])
        b_t = np.random.normal(0., sigma, dim)
        # print(np.linalg.norm(w_t))
        # print(np.linalg.norm(b_t))
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
    options['epsilon'] = .2

    options['n_tr'] = 256
    options['n_repeat'] = 5
    options['delta'] = 1. / options['n_tr']
    # options['n_pass'] = np.log2(options['n_tr'] ** 2).astype(int) + 1
    options['n_pass'] = options['n_tr']
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
        auc_t, time_t = auc_dp_egd_logistic(x_tr, y_tr, x_te, y_te, options)
        time_rec[i] = time_t
        auc_rec[i] = auc_t
    print('----------------------')
    print(np.mean(auc_rec, axis=0))
    print(np.std(auc_rec, axis=0))