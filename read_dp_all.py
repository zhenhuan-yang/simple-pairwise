from read_res import read_dp_res
from itertools import product

# method_list = ['fifo', 'olp', 'pair', 'oam_gra']

# data_list = ['australian', 'diabetes', 'dna', 'fourclass', 'german', 'heart', 'ionosphere', 'iris',
#              'letter', 'liver-disorders', 'pendigits', 'satimage', 'sonar', 'splice', 'svmguide1', 'usps', 'vehicle',
#              'vowel', 'wine']

data_list = ['diabetes', 'german']
#
# for data, method in product(data_list, method_list):
#     method_name = 'auc_' + method
#     read_res(data, 'hinge', method_name)

epsilon_list = [1., 1.5, 2., 5.]
method_list = ['fifo', 'egd', 'offpairc']

for data, method, epsilon in product(data_list, method_list, epsilon_list):
    method_name = 'auc_dp_' + method
    read_dp_res(data, 'logistic', method_name, epsilon)