import math
import random
import copy

discretization = 1
function_dimensions = {
    'fct': 1
}
function_domains = {
    'fct': [(0, 31)]
}


def fct(x):
    return x[0] ** 3 - 60 * x[0] ** 2 + 900 * x[0] + 100


functions = {
    'fct': fct
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
    return {'values': [random.randint(0, 1) for i in range(chromosome_size)], 'dimensions': dimensions}


def get_normal_values(chromosome, function_name):
    x = []
    past_position = 0
    for i in range(function_dimensions[function_name]):
        new_position = past_position + chromosome['dimensions'][i]
        base_10_number = int(''.join(str(i) for i in chromosome['values'][past_position:new_position]), 2)
        past_position = new_position
        domain = function_domains[function_name][i]
        x.append(base_10_number/(2**chromosome['dimensions'][i] - 1)*(domain[1] - domain[0]) + domain[0])
    return x


def apply_function(chromosome, function_name):
    real_values = get_normal_values(chromosome, function_name)
    return functions[function_name](real_values)


def flip_bit(bit):
    return (bit + 1) % 2


def flip_bit_from_list(bit_list, position):
    bit_list[position] = flip_bit(bit_list[position])
    return bit_list


def hill_climbing(function_name, iterations=1000, best_improvement=False):
    iteration_number = 0
    chromosome = initialize_chromosome(function_name)
    best_fitness = apply_function(chromosome, function_name)
    temp_chromosome = None
    temp_best_fitness = best_fitness
    best_value = get_normal_values(chromosome, function_name)[0]
    current_progress = [best_value]
    temp_chromosome = chromosome
    # print('initial fitness', best_fitness)
    while True:
        if iteration_number == iterations:
            return best_value, current_progress
        iteration_number += 1
        # shuffled_indexes = random.sample(list(range(len(chromosome['values']))), len(chromosome['values']))
        shuffled_indexes = range(len(chromosome['values']) - 1, -1, -1)
        for i in shuffled_indexes:
            neighbour = copy.deepcopy(chromosome)
            neighbour['values'] = flip_bit_from_list(neighbour['values'], i)
            current_fitness = apply_function(neighbour, function_name)
            if current_fitness > temp_best_fitness:
                temp_chromosome = neighbour
                temp_best_fitness = current_fitness
                # print(temp_best_fitness)
                if not best_improvement:
                    break
        # if best_fitness == temp_best_fitness:
            # break
        chromosome = temp_chromosome
        best_fitness = temp_best_fitness
        best_value = get_normal_values(chromosome, function_name)[0]
        if best_value not in current_progress:
            current_progress.append(best_value)


def restart_hill_climbing(restarts=100, best_improvement=True):
    bazin = {}
    for _ in range(restarts):
        a = hill_climbing('fct', iterations=100, best_improvement=best_improvement)
        init_bazin = bazin.get(a[0], [])
        new_bazin = set(init_bazin + a[1])
        bazin[a[0]] = list(new_bazin)
    for i in bazin:
        print(i, bazin[i])


print('Best improvement')
restart_hill_climbing()
print('First improvement')
restart_hill_climbing(best_improvement=False)
