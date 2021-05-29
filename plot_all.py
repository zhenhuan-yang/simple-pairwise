from plot_command import plot_command

# data_list = ['australian', 'diabetes', 'dna', 'fourclass', 'german', 'heart', 'ionosphere', 'iris',
#              'letter', 'liver-disorders', 'pendigits', 'satimage', 'sonar', 'splice',
#              'svmguide1', 'usps', 'vehicle', 'vowel', 'wine']
# data_list = ['a9a', 'ijcnn1', 'mnist', 'w8a']
data_list = ['diabetes', 'german', 'ijcnn1', 'letter', 'mnist', 'usps']
for data in data_list:
    plot_command(data, 'hinge', True)
    plot_command(data, 'square', True)
    plot_command(data, 'loglink', True)
