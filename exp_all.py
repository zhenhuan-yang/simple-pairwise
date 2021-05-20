from exp_command import exp_command
from itertools import product

data_list = ['colon-cancer', 'mnist', 'gisette_scale', 'madelon', 'satimage', 'splice', 'letter', 'mushroom', 'ijcnn1', 'w8a', 'a9a', 'svmguide3']
method_list = ['auc_fifo', 'auc_pair', 'auc_oam_gra', 'auc_olp', 'auc_oam_gra_1', 'auc_olp_1']

for data, method in product(data_list, method_list):
    exp_command(data, 'hinge', method)

for data in data_list:
    exp_command(data, 'square', 'auc_spauc')