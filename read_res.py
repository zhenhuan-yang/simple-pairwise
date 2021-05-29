import os
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt

def read_res(data, loss, method):

    # load results
    cur_path = os.getcwd()
    res_path = os.path.join(cur_path, 'res')
    filename = data + '_' + method + '_' + loss
    with open(os.path.join(res_path, filename + '.pickle'), 'rb') as rfile:
        res = pkl.load(rfile)

    print('------- %s -------' % data)
    time = res['time']
    auc = res['auc']
    mean_str = str(np.max(auc['mean'])).lstrip('0')[:4]
    std_str = str(np.min(auc['std'])).lstrip('0')[:4]
    print('method: %s auc: $%s \pm %s$' %(method, mean_str, std_str))

    return

def read_dp_res(data, loss, method, epsilon):

    # load results
    cur_path = os.getcwd()
    res_path = os.path.join(cur_path, 'res')
    filename = data + '_' + method + '_' + loss + '_eps_' + str(epsilon)
    with open(os.path.join(res_path, filename + '.pickle'), 'rb') as rfile:
        res = pkl.load(rfile)

    print('------- %s -------' % data)
    time = res['time']
    auc = res['auc']
    mean_str = str(np.max(auc['mean'])).lstrip('0')[:4]
    std_str = str(np.min(auc['std'])).lstrip('0')[:4]
    print('method: %s eps: %.1f auc: $%s \pm %s$' %(method, epsilon, mean_str, std_str))

    return

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Read results')
    parser.add_argument('-d', '--data', default='diabetes')
    parser.add_argument('-m', '--method', default='fifo')
    parser.add_argument('-l', '--loss', default='square')
    args = parser.parse_args()
    method_name = 'auc_' + args.method
    read_res(args.data, args.loss, method_name)