import os
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt

def plot_dp_res(data, loss, epsilon, method_list, legend_dict, color_list, line_list, flag_time):

    # Using seaborn's style
    plt.style.use('seaborn')
    width = 345
    tex_fonts = {
        # Use LaTeX to write all text
        "text.usetex": True,
        "font.family": "serif",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 16,
        "font.size": 16,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 12,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16
    }
    plt.rcParams.update(tex_fonts)
    fig = plt.figure(figsize=(8, 6))

    # load results
    cur_path = os.getcwd()
    res_path = os.path.join(cur_path, 'res')
    res = dict()

    for method in method_list:
        filename = data + '_' + method + '_' + loss + '_eps_' + str(epsilon)
        with open(os.path.join(res_path, filename + '.pickle'), 'rb') as rfile:
            res[method] = pkl.load(rfile)

    print('------- %s -------' % data)
    # time chop
    max_time = 0.
    for i, method in enumerate(method_list):
        time = res[method]['time']
        auc = res[method]['auc']
        if time['mean'][-1] > max_time:
            max_time = time['mean'][-1]
    min_time = 1e8
    for i, method in enumerate(method_list):
        time = res[method]['time']
        auc = res[method]['auc']
        if time['mean'][0] < min_time:
            min_time = time['mean'][0]
    max_shift = 0.
    for i, method in enumerate(method_list):
        time = res[method]['time']
        auc = res[method]['auc']
        # reciprocal = time['mean'][0] / np.linspace(time['mean'][0], time['mean'][-1], num=len(time['mean'])) + (
        #             1.1 * (1 - auc['mean'][-1]) - time['mean'][0] / time['mean'][-1])
        # plt.plot(time['mean'][1:], reciprocal[1:],  linewidth=2, color='yellowgreen', label=r'c_1/(x+c_2)')
        if flag_time:
            if time['mean'][0] - min_time > max_shift:
                max_shift = time['mean'][0] - min_time
            plt.plot(time['mean'] - (time['mean'][0] - min_time), auc['mean'], color=color_list[i], linewidth=1.5, marker=line_list[i],
                     markeredgewidth=1, markersize=5, label=legend_dict[method])
            # plt.plot(time['mean'][mask], auc['mean'][mask], color=color_list[i], linewidth=1.5, marker=line_list[i],
            #          markeredgewidth=1, markersize=5, label=legend_dict[method])
            # plt.errorbar(time['mean'][mask], auc['mean'][mask], yerr=auc['std'][mask] / 4, color=color_list[i], linewidth=1.5,
            #              fmt=line_list[i], capsize=3, elinewidth=1, markeredgewidth=1, markersize=5, label=method)
            plt.xlabel(r'\textbf{CPU running time ($\log$ scale)}', fontsize=24)
        else:
            iters = np.arange(1, len(auc['mean']) + 1 )
            plt.errorbar(iters, auc['mean'], yerr=auc['std'] / 4, color=color_list[i],
                         linewidth=1.5,
                         fmt=line_list[i], capsize=3, elinewidth=1, markeredgewidth=1, markersize=5, label=method)
            plt.xlabel('log iter')
    # non-private
    n_idx = len(auc['mean'])
    filename = data + '_auc_fifo_' + loss
    with open(os.path.join(res_path, filename + '.pickle'), 'rb') as rfile:
        baseline = pkl.load(rfile)

    time = baseline['time']
    auc = baseline['auc']
    flat_auc = np.ones(n_idx) * np.max(auc['mean'])
    plt.plot(np.linspace(min_time, max_time - max_shift, n_idx), flat_auc, color='darkgreen', linewidth=2,
             linestyle='dashed', label=r'\texttt{Non-Private}')
    plt.ylabel(r'\textbf{AUC score}', fontsize=24)
    plt.legend(loc='lower right', prop={'size': 16})
    plt.title(r'$\epsilon = %s$' % str(epsilon), fontsize=24)

    return fig

if __name__ == '__main__':
    loss = 'logistic'
    method_list = ['auc_dp_fifo', 'auc_dp_egd', 'auc_dp_offpairc']
    legend_list = [r'\texttt{Ours}', r'$\texttt{DPEGD}$', r'\texttt{OffPairC}']
    legend_dict = dict(zip(method_list, legend_list))
    data = 'german'
    # name_list = ['Simple SGD']
    color_list = ['navy', 'maroon', 'orange']
    marker_list = ['o', '>', '*']
    line_list = ['--o', '-->', '--*']
    fig = plot_dp_res(data, loss, 2., method_list, legend_dict, color_list, marker_list, True)
    plt.show()
    plt.savefig('fig/' + data + '_' + loss + '.png', dpi=600, bbox_inches='tight', pad_inches=0.05)

