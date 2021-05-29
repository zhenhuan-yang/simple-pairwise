from exp_dp_command import exp_dp_command
from itertools import product

data_list = ['diabetes', 'german']
epsilon_list = [1., 1.5, 2., 5.]
# epsilon_list = [.2, .5, .8]
method_list = ['fifo', 'egd', 'offpairc']

for data, method, epsilon in product(data_list, method_list, epsilon_list):
    exp_dp_command(data, 'logistic', method, epsilon)
#
# for data in data_list:
#     exp_dp_command(data, 'logistic', 'non-private', 1.)
