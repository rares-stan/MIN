import math
from functools import reduce
import random
import copy

discretization = 100
function_dimensions = {
    'rastrigin': 30,
    'griewangk': 2,
    'rosenbrock': 2,
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


def calculate_variable_space(domain):
    return math.ceil(math.log((discretization * domain[1] - domain[0]), 2))


def calculate_chromosome_size(function_name):
    size = 0
    variable_sizes = []
    for domain in function_domains[function_name]:
        variable_space = calculate_variable_space(domain)
        size += variable_space
        variable_sizes.append(variable_space)
    return size, variable_sizes


def initialize_chromosome(function_name):
    chromosome_size, dimensions = calculate_chromosome_size(function_name)
    return {'value': [random.randint(0, 1) for i in range(chromosome_size)], 'dimensions': dimensions}


def get_normal_values(chromosome, function_name):
    x = []
    past_position = 0
    for i in range(function_dimensions[function_name]):
        new_position = past_position + chromosome['dimensions'][i]
        base_10_number = int(''.join(str(i) for i in chromosome['value'][past_position:new_position]), 2)
        past_position = new_position
        domain = function_domains[function_name][i]
        x.append(base_10_number/(2**chromosome['dimensions'][i] - 1)*(domain[1] - domain[0]) + domain[0])
    return x


def check_fitness(chromosome, function_name):
    real_values = get_normal_values(chromosome, function_name)
    return functions[function_name](real_values)


def hillclimbing(function_name, iterations=1000, best_improvement=False):
    iteration_number = 0
    chromosome = initialize_chromosome(function_name)
    best_fitness = check_fitness(chromosome, function_name)
    temp_chromosome = None
    temp_best_fitness = best_fitness
    print('initial fitness', best_fitness)
    while True:
        if iteration_number == iterations:
            break
        iteration_number += 1
        # shuffled_indexes = random.sample(list(range(len(chromosome['value']))), len(chromosome['value']))
        shuffled_indexes = range(len(chromosome['value']))
        for i in shuffled_indexes:
            neighbour = copy.deepcopy(chromosome)
            neighbour['value'][i] = (neighbour['value'][i] + 1) % 2
            current_fitness = check_fitness(neighbour, function_name)
            if current_fitness < temp_best_fitness:
                temp_chromosome = neighbour
                temp_best_fitness = current_fitness
                print(temp_best_fitness)
                if not best_improvement:
                    break
        if best_fitness == temp_best_fitness:
            break
        chromosome = temp_chromosome
        best_fitness = temp_best_fitness
    return chromosome, best_fitness


for restart in range(1):
    name = 'rastrigin'
    # name = 'griewangk'
    # name = 'rosenbrock'
    # name = 'six_hump'
    print('First improvement')
    val = hillclimbing(name, iterations=1000)
    print(val)
    print(get_normal_values(val[0], name), val[1])
    print('------------------------------------------------------------------')
    print('Best improvement')
    val = hillclimbing(name, iterations=1000, best_improvement=True)
    print(val)
    print(get_normal_values(val[0], name), val[1])
