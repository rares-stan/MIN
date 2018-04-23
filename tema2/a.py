from functools import reduce
import numpy as np
import math
import random

function_dimensions = {
    'rastrigin': 30,
    'griewangk': 30,
    'rosenbrock': 30,
    'six_hump': 2
}
function_domains = {
    'rastrigin': [(-5.12, 5.12)] * function_dimensions['rastrigin'],
    'griewangk': [(-600, 600)] * function_dimensions['griewangk'],
    'rosenbrock': [(-2.048, 2.048)] * function_dimensions['rosenbrock'],
    'six_hump': [(-3, 3), (-2, 2)]
}


def rastrigin(x):
    return reduce(lambda acc, xi: acc + (xi ** 2 - (10 * math.cos(2 * math.pi * xi))), x, 10 * len(x))


def griewangk(x):
    function_sum = reduce(lambda acc, xi: acc + (xi ** 2 / 4000), x, 0)
    function_product = 1
    for i in range(len(x)):
        function_product *= math.cos(x[i] / math.sqrt(i + 1))
    return function_sum - function_product + 1


def rosenbrock(x):
    function_sum = 0
    for index in range(len(x) - 1):
        function_sum += (100 * (x[index + 1] - x[index] ** 2) ** 2 + (1 - x[index]) ** 2)
    return function_sum


def six_hump(x):
    return (4-2.1*x[0]**2+(x[0]**4)/3)*(x[0]**2) + x[0]*x[1] + (-4 + 4*x[1]**2)*(x[1]**2)


functions = {
    'rastrigin': rastrigin,
    'griewangk': griewangk,
    'rosenbrock': rosenbrock,
    'six_hump': six_hump
}


def pso_init_population(pop_size, vmax, function_name):
    current_domain = function_domains[function_name]
    dimension = function_dimensions[function_name]
    x = np.array([
        np.array([
            random.uniform(current_domain[i][0], current_domain[i][1])
            for i in range(dimension)
        ])
        for _ in range(pop_size)
    ])
    v = np.array([
        np.array([
            random.uniform(-vmax[i], vmax[i])
            for i in range(dimension)
        ])
        for _ in range(pop_size)
    ])
    return x, v


def pso_update_particle(current_x, current_v, vmax, global_best, particle_best, w):
    new_velocity = w[0]*current_v + w[1]*random.random()*(global_best - current_x) + w[2]*random.random()*(particle_best - current_x)
    new_velocity = np.array(list(map(lambda x: x[1] if x[1] <= vmax[x[0]] else vmax[x[0]], enumerate(new_velocity))))
    new_x = current_x + new_velocity
    return new_x, new_velocity


def pso_update_population(x, v, vmax, global_best, particle_best, w, function_name):
    new_x = np.array([None]*len(x))
    new_v = np.array([None]*len(v))
    for i in range(len(x)):
        new_x[i], new_v[i] = pso_update_particle(x[i], v[i], vmax, global_best, particle_best[i], w)
    return new_x, new_v


def pso_evaluate_population(x, function_name):
    global current_eval
    current_eval += len(x)
    return list(map(functions[function_name], x))


def pso(function_name, w, vmax, pop_size, max_iterations, max_eval):
    global_best = None
    particle_best = [None]*pop_size
    x, v = pso_init_population(pop_size, vmax, function_name)
    iterations = 0
    while current_eval < max_eval:
        iterations += 1
        results = pso_evaluate_population(x, function_name)
        current_best = (results.index(min(results)), min(results))
        global_best = global_best if global_best and global_best[1] < current_best[1] else (x[current_best[0]], current_best[1])
        for i in range(len(results)):
            particle_best[i] = particle_best[i] if particle_best[i] and particle_best[i][1] < results[i] else (x[i], results[i])
        x, v = pso_update_population(x, v, vmax, global_best[0], list(zip(*particle_best))[0], w, function_name)
        w[0] = w[0] - 0.9/max_iterations
    print(global_best[1])


for _ in range(30):
    current_eval = 0
    # current_function = 'rastrigin'
    # current_function = 'griewangk'
    current_function = 'rosenbrock'
    # current_function = 'six_hump'
    hyper_w = [0.9, 2, 2]
    hyper_vmax = list(zip(*function_domains[current_function]))[1]
    pso(current_function, hyper_w, hyper_vmax, 20, 1000, max_eval=100000)
