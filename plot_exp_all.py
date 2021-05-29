'''
dataset n, d
a9a 32561, 123
australian 690, 14
breast-cancer* 683, 10
cifar10 50000, 3072
cod-rna 59535, 8
colon-cancer* 62, 2000
connect-4 67557, 126
covtype* 581012, 54
diabetes 768, 8
dna 2000, 180
fourclass 862, 2
german 1000, 24
gisette* 6000, 5000
glass* 214, 9
heart 270, 13
ijcnn1 49990, 22
ionosphere 351, 34
iris 150, 4
letter 15000, 161
leukemia* 38, 7129li
liver-disorders 145, 5
madelon* 2000, 500
mnist 60000, 780
mushromms 8124, 112
pendigits 7494, 16
phishing 11055 68
poker* 25010, 10
protein* 17766, 357
satimage 4435, 36
segment* 2310, 19
sensorless 58509, 48
shuttle 43500, 9
skin_nonskin 245057, 3
sonar 208, 60
splice 1000, 60
svmguide1 3089, 4
svmguide3* 1243, 21
usps 7291, 256
vehicle 846, 18
vowel 528, 10
w8a 49749, 300
wine 178, 13
'''

from plot_exp_command import plot_exp_command
from itertools import product

# data_list = ['a9a', 'ijcnn1', 'mnist', 'w8a']


# data_list = ['australian', 'diabetes', 'dna', 'fourclass', 'german', 'heart', 'ionosphere', 'iris',
#              'letter', 'liver-disorders', 'pendigits', 'satimage', 'sonar', 'splice',
#              'svmguide1', 'usps', 'vehicle', 'vowel', 'wine']

data_list = ['diabetes', 'german', 'ijcnn1', 'letter', 'mnist', 'usps']
method_list = ['fifo']

for data, method in product(data_list, method_list):
    plot_exp_command(data, 'loglink', method)

# method_list = ['fifo', 'pair']
#
# for data, method in product(data_list, method_list):
#     plot_exp_command(data, 'square', method)
