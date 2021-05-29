from plot_res import plot_res

def plot_command(data, loss, flag_time):
    if loss == 'hinge':
        method_list = ['auc_fifo', 'auc_pair', 'auc_olp', 'auc_oam_gra']
        legend_list = [r'\texttt{Ours}', r'$\texttt{SGD}_{pair}$', r'\texttt{OLP}', r'$\texttt{OAM}_{gra}$']
        legend_dict = dict(zip(method_list, legend_list))
        color_list = ['navy', 'maroon', 'orange', 'yellowgreen']
        line_list = ['--o', '-->', '--*', '--s']
        marker_list = ['o', '>', '*', 's']
    elif loss == 'square':
        method_list = ['auc_fifo', 'auc_pair', 'auc_olp', 'auc_spauc']
        legend_list = [r'\texttt{Ours}', r'$\texttt{SGD}_{pair}$', r'\texttt{OLP}', r'\texttt{SPAUC}']
        legend_dict = dict(zip(method_list, legend_list))
        color_list = ['navy', 'maroon', 'orange', 'yellowgreen']
        line_list = ['--o', '-->', '--*', '--s']
        marker_list = ['o', '>', '*', 's']
    elif loss == 'loglink':
        method_list = ['auc_fifo']
        legend_list = [r'\texttt{Ours}']
        legend_dict = dict(zip(method_list, legend_list))
        color_list = ['navy']
        line_list = ['--o']
        marker_list = ['o']

    fig = plot_res(data, loss, method_list, legend_dict, color_list, marker_list, flag_time)
    if flag_time:
        fig.savefig('fig/' + data + '_' + loss + '_time' + '.png', dpi=600, bbox_inches='tight', pad_inches=0.05)
    else:
        fig.savefig('fig/' + data + '_' + loss + '.png', dpi=600, bbox_inches='tight', pad_inches=0.05)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Training setting')
    parser.add_argument('-d', '--data', default='diabetes')
    parser.add_argument('-t', '--time', action='store_true')
    args = parser.parse_args()

    plot_command(args.data, args.time)