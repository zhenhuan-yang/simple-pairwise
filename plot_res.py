import os
import pickle as pkl
import numpy as np
import matplotlib.pyplot as plt

def plot_res(data, loss, method_list, legend_dict, color_list, line_list, flag_time):

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
        filename = data + '_' + method + '_' + loss + '_4plot'
        with open(os.path.join(res_path, filename + '.pickle'), 'rb') as rfile:
            res[method] = pkl.load(rfile)

    print('------- %s -------' % data)
    # time chop
    min_time = 1e8
    for i, method in enumerate(method_list):
        time = res[method]['time']
        auc = res[method]['auc']
        if time['mean'][-1] < min_time:
            min_time = time['mean'][-1]

    for i, method in enumerate(method_list):
        time = res[method]['time']
        auc = res[method]['auc']
        # reciprocal = time['mean'][0] / np.linspace(time['mean'][0], time['mean'][-1], num=len(time['mean'])) + (
        #             1.1 * (1 - auc['mean'][-1]) - time['mean'][0] / time['mean'][-1])
        # plt.plot(time['mean'][1:], reciprocal[1:],  linewidth=2, color='yellowgreen', label=r'c_1/(x+c_2)')
        if flag_time:
            mask = time['mean'] < min_time
            plt.plot(time['mean'], auc['mean'], color=color_list[i], linewidth=1.5, marker=line_list[i],
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
    plt.ylabel(r'\textbf{AUC score}', fontsize=24)
    plt.xscale('log')
    plt.legend(loc='lower right', prop={'size': 16})
    plt.title(r'\texttt{%s}' % data, fontsize=24)

    return fig

if __name__ == '__main__':
    loss = 'hinge'
    method_list = ['auc_fifo', 'auc_pair', 'auc_olp', 'auc_oam_gra']
    legend_list = [r'\texttt{Ours}', r'$\texttt{SGD}_{pair}$', r'\texttt{OLP}', r'$\texttt{OAM}_{gra}$']
    legend_dict = dict(zip(method_list, legend_list))
    data = 'usps'
    # name_list = ['Simple SGD']
    color_list = ['navy', 'maroon', 'orange', 'yellowgreen']
    marker_list = ['o', '>', '*', 's']
    line_list = ['--o', '-->', '--*', '--s']
    fig = plot_res(data, loss, method_list, legend_dict, color_list, marker_list, True)
    plt.show()
    plt.savefig('fig/' + data + '_' + loss + '.png', dpi=600, bbox_inches='tight', pad_inches=0.05)

