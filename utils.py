import numpy as np
from scipy import io as spio
import scipy.sparse as sp
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, Normalizer, StandardScaler, RobustScaler
from sklearn.datasets import load_svmlight_file
import os


def data_processing(data):
    # ------------------------

    # SGD for AUC optimization without regularization

    cur_path = os.getcwd()
    if os.path.exists(os.path.join(cur_path, 'data')):
        data_path = os.path.join(cur_path, 'data')
    else:
        data_path = os.path.join(os.path.dirname(cur_path), 'data')
    # -----------------------------------------
    # processing the data
    print(data)
    if data == 'rcv1' or data == 'gisette' or data == 'madelon':
        x_tr, y_tr = load_svmlight_file(os.path.join(data_path, data))
        x_te, y_te = load_svmlight_file(os.path.join(data_path, data) + '.t')

        #        n_tr, n_te = len(y_tr), len(y_te)
        x = sp.vstack((x_tr, x_te))
        # x = x.toarray()
        y = np.hstack((y_tr, y_te))
    elif data == 'CCAT' or data == 'astro' or data == 'cov1':
        tmp = spio.loadmat(os.path.join(data_path, data))
        x, y = tmp['Xtrain'], tmp['Xtest']
    elif data == 'protein_h':
        data_tr = np.loadtxt(os.path.join(data_path, data))
        #        data_te = np.loadtxt('data/'+data+'_t')
        x, y = data_tr[:, 3:], data_tr[:, 2]
        x = MinMaxScaler.fit_transform(x)
    #        x_te, y_te = data_te[:,3:], data_te[:,2]
    #        x = sp.vstack((x_tr, x_te))
    #        y = np.hstack((y_tr, y_te))
    elif data == 'smartBuilding' or data == 'malware':
        data_ = spio.loadmat(os.path.join(data_path, data))['data']
        x = data_[:, 1:]
        x = MinMaxScaler.fit_transform(x)
        y = data_[:, 0]
    elif data == 'http' or data == 'smtp' or data == 'shuttle' or data == 'cover':
        tmp = spio.loadmat(os.path.join(data_path, data))
        x = tmp['X']
        y = tmp['y'].ravel()
        y = np.int16(y)
        x = MinMaxScaler.fit_transform(x)
        tmp = []
    # elif data == 'diabetes' or data == 'heart':
    #     x, y = load_svmlight_file(os.path.join(data_path, data))
    #     x = x.toarray()
    #     x = MinMaxScaler(feature_range=(-1, 1)).fit_transform(x)
    else:
        x, y = load_svmlight_file(os.path.join(data_path, data))
        x = x.toarray()
        x = MinMaxScaler(feature_range=(-1, 1)).fit_transform(x)

    n_data, n_dim = x.shape
    print('(n,dim)=(%s,%s)' % (n_data, n_dim))
    # check the categories of the data, if it is the multi-class data, process it to binary-class
    uLabel = np.unique(y)
    print('uLabel = %s' % uLabel)
    uNum = len(uLabel)
    if uNum == 2:
        y[y != 1] = -1
    if uNum > 2:
        #        uSort = np.random.permutation(uNum)
        #        uSort = np.arange(uNum)
        ty = y
        y = np.ones(n_data, dtype=int)
        for k in np.arange(int(uNum / 2), uNum, dtype=int):  # negative class
            #            print(uLabel[k])
            y[ty == uLabel[k]] = -1

    return x, y

def get_idx(n_data, n_pass):
    # idx = np.zeros(n_data * n_pass, dtype=int)
    # random permutation
    # for i_pass in np.arange(n_pass):
        # idx[i_pass * n_data : (i_pass + 1) * n_data] = np.random.permutation(n_data)
    # random selection
    # for i in range(n_data * n_pass):
    #     idx[i] = np.random.randint(n_data)
    idx = np.random.randint(n_data, size=n_data * n_pass)
    return idx

# def get_batch_idx(n_data, n_pass, options):
#     n_iter = int(n_data * n_pass / options['batch_size'])
#     idx = np.zeros((n_iter, options['batch_size']), dtype=int)
#     for i in range(n_iter):
#         idx[i] = np.random.choice(n_data, options['batch_size'], replace=False)
#     return idx

def get_past_idx(n_data, n_pass):
    # idx = np.zeros((n_data * n_pass, 2), dtype=int)
    # # random permutation
    # # for i_pass in np.arange(n_pass):
    #     # idx[i_pass * n_data : (i_pass + 1) * n_data] = np.random.permutation(n_data)
    # # random selection
    # for i in range(n_data * n_pass):
    #     idx[i] = np.random.choice(n_data, size=2, replace=False)
    idx = np.zeros((n_data * n_pass, 2), dtype=int)
    for i in range(n_data * n_pass):
        idx[i][0] = np.random.randint(n_data)
        idx[i][1] = idx[i-1][0]
    return idx

def get_pair_idx(n_data, n_pass):
    idx = np.zeros((n_data * n_pass, 2), dtype=int)
    # # random permutation
    # # for i_pass in np.arange(n_pass):
    #     # idx[i_pass * n_data : (i_pass + 1) * n_data] = np.random.permutation(n_data)
    # # random selection
    # for i in range(n_data * n_pass):
    #     idx[i] = np.random.choice(n_data, size=2, replace=False)
    # idx = np.zeros((n_data * n_pass, 2), dtype=int)
    # for i in range(n_data * n_pass):
    #     idx[i][0] = np.random.randint(n_data)
    #     # arr = np.arange(n_data)
    #     # mask = np.ones(len(arr), dtype=bool)
    #     # mask[idx[i][0]] = False
    #     # after_arr = arr[mask]
    #     # idx[i][1] = np.random.choice(after_arr)
    #     idx[i][1] = np.random.randint(n_data)
    idx1= np.random.randint(n_data, size=n_data * n_pass)
    idx2 = np.random.randint(n_data - 1, size=n_data * n_pass)
    I = np.where(idx1 == idx2)
    idx2[I[0]] = n_data - 1
    idx[:, 0] = idx1
    idx[:, 1] = idx2
    return idx

def get_res_idx(n_iter, options):
    if options['log_res']:
        res_idx = (2 ** (np.arange(4, np.log2(n_iter), options['rec_log']))).astype(int)
    else:
        res_idx = np.arange(1, n_iter, options['rec'])
    res_idx[-1] = n_iter
    # res_idx = [int(i) for i in res_idx] # map(int, res_idx)
    return res_idx

def get_stage_idx(n_tr):
    stage_ids = (n_tr * 0.5 ** (np.arange(1, np.log2(n_tr)))).astype(int)
    stages = []
    count = 0
    for stage_idx in stage_ids:
        stages.append(range(count, count + stage_idx))
        count = count + stage_idx
    return stages

def get_stage_res_idx(stages, n_pass):
    res_idx = []
    for stage in stages:
        res_idx.append(stage[-1] * n_pass)
    return res_idx

def get_etas(n_iter, eta, options):
    if options['eta_geo'] == 'const':
        etas = eta * np.ones(n_iter) / np.sqrt(n_iter)
    elif options['eta_geo'] == 'sqrt':
        etas = eta / np.sqrt(np.arange(1, n_iter + 1))
    elif options['eta_geo'] == 'fast':
        etas = eta / np.arange(1, n_iter + 1)
    else:
        print('Wrong step size geometry!')
        etas = eta * np.ones(n_iter) / np.sqrt(n_iter)
    return etas