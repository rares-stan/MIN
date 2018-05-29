import numpy as np
import scipy.stats as st


def calculate_things(data):
    print('min', data.min())
    print('max', data.max())
    print('dev_std', np.std(data))
    print('medie', data.mean())
    print('interval incredere', st.t.interval(0.95, len(data)-1, loc=np.mean(data), scale=st.sem(data)))


def read_data(filename):
    with open(filename, 'r') as f:
        data = np.array(list(map(lambda x: float(x), f.readlines())))
    return data


approaches = ['ga', 'gah', 'pso']

files = [
    'griewangk-10.txt',
    'griewangk-30.txt',
    'rastrigin-10.txt',
    'rastrigin-30.txt',
    'rosenbrock-10.txt',
    'rosenbrock-30.txt',
    'six_hump.txt'
]


for approach in approaches:
    for file in files:
        file_str = '%s/%s-%s' % (approach, approach, file)
        print(file_str)
        current_data = read_data(file_str)
        calculate_things(current_data)
        print()
        print()

