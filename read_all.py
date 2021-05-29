from read_res import read_res
from itertools import product

method_list = ['olp_1', 'oam_gra_1']

# data_list = ['australian', 'diabetes', 'dna', 'fourclass', 'german', 'heart', 'ionosphere', 'iris',
#              'letter', 'liver-disorders', 'pendigits', 'satimage', 'sonar', 'splice', 'svmguide1', 'usps', 'vehicle',
#              'vowel', 'wine']

data_list = ['diabetes', 'german', 'ijcnn1', 'letter', 'mnist', 'usps']
#
# for data, method in product(data_list, method_list):
#     method_name = 'auc_' + method
#     read_res(data, 'hinge', method_name)
#
for data, method in product(data_list, method_list):
    method_name = 'auc_' + method
    read_res(data, 'hinge', method_name)