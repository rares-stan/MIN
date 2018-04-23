import math
from functools import reduce
import random
import copy
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

discretization = 100
function_dimensions = {
    'rastrigin': 10,
    'griewangk': 10,
    'rosenbrock': 10,
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


def hill_climbing(function_name, max_eval, start_chromosome=None, iterations=1000, best_improvement=False):
    global eval_calls
    iteration_number = 0
    chromosome = start_chromosome if start_chromosome else initialize_chromosome(function_name)
    best_fitness = apply_function(chromosome, function_name)
    temp_chromosome = None
    temp_best_fitness = best_fitness
    # print('initial fitness', best_fitness)
    while True:
        if iteration_number == iterations:
            break
        iteration_number += 1
        # shuffled_indexes = random.sample(list(range(len(chromosome['values']))), len(chromosome['values']))
        shuffled_indexes = range(len(chromosome['values']))
        for i in shuffled_indexes:
            neighbour = copy.deepcopy(chromosome)
            neighbour['values'] = flip_bit_from_list(neighbour['values'], i)
            eval_calls += 1
            if eval_calls > max_eval:
                return chromosome, best_fitness
            current_fitness = apply_function(neighbour, function_name)
            if current_fitness < temp_best_fitness:
                temp_chromosome = neighbour
                temp_best_fitness = current_fitness
                # print(temp_best_fitness)
                if not best_improvement:
                    break
        if best_fitness == temp_best_fitness:
            break
        chromosome = temp_chromosome
        best_fitness = temp_best_fitness
    return chromosome, best_fitness


def mutate_population(population, mutation_chance):
    return list(
        map(
            lambda chromosome:
                list(map(
                    lambda gene:
                    flip_bit(gene) if random.random() < mutation_chance else gene,
                    chromosome
                )),
            population
        )
    )


def crossover(p1, p2):
    position = random.randint(1, len(p1) - 2)
    o1 = p1[:position] + p2[position:]
    o2 = p2[:position] + p1[position:]
    return o1, o2


def get_crossover_populations(population, crossover_chance):
    pop_probability = list(zip(population, [random.random() for _ in range(len(population))]))
    pop_for_crossover = list(filter(lambda p: p[1] < crossover_chance, pop_probability))
    pop_for_crossover = sorted(pop_for_crossover, key=lambda x: x[1])
    rest = list(filter(lambda x: x[1] >= crossover_chance, pop_probability))
    if len(pop_for_crossover) % 2 == 1:
        if random.random() > 0.5:
            prob_min = min(rest, key=lambda x: x[1])
            pop_for_crossover.append(prob_min)
            rest.remove(prob_min)
        else:
            rest.append(pop_for_crossover.pop())
    rest = list(list(zip(*rest))[0])
    pop_for_crossover, prob = zip(*pop_for_crossover)
    p1 = [pop_for_crossover[i] for i in range(0, len(pop_for_crossover), 2)]
    p2 = [pop_for_crossover[i] for i in range(1, len(pop_for_crossover), 2)]
    return p1, p2, rest


def crossover_population(population, crossover_chance):
    p1, p2, rest = get_crossover_populations(population, crossover_chance)
    pop = list(zip(p1, p2))
    pop = list(map(lambda x: crossover(*x), pop))
    p1, p2 = zip(*pop)
    return list(p1) + list(p2) + rest


def hybridise_hill_climbing(population, function_name, mutate_chance, max_eval):
    return list(
        map(
            lambda chromosome:
                hill_climbing(
                    function_name,
                    max_eval,
                    {'values': chromosome, 'dimensions': population['dimensions']},
                    iterations=1,
                    best_improvement=False
                )[0]['values'] if
                random.random() < mutate_chance else
                chromosome,
            population['values']
        )
    )


def initialize_population(function_name, pop_size):
    dimensions = initialize_chromosome(function_name)['dimensions']
    return {'values': [initialize_chromosome(function_name)['values'] for _ in range(pop_size)], 'dimensions': dimensions}


def evaluate_population(population, function_name):
    global eval_calls
    eval_calls += len(population['values'])
    return list(map(lambda x: apply_function({'values': x, 'dimensions': population['dimensions']}, function_name), population['values']))


def calculate_fitness(results):
    c = max(results) + 1
    return map(lambda x: -1 * x + c, results)


def calculate_chances(fitness, total):
    chances = [fitness[0]/total]
    for i in range(1, len(fitness)):
        chances.append(chances[-1] + fitness[i]/total)
    return chances


def find_min_grater(chances, number):
    for i in range(0, len(chances)):
        if chances[i] > number:
            return i


def generation_selection(population, fitness):
    total_fitness = sum(fitness, 0)
    chances = calculate_chances(fitness, total_fitness)
    new_pop = list(
        map(
            lambda x: population['values'][find_min_grater(chances, x)],
            [random.random() for _ in range(len(fitness))]
        )
    )
    return {'values': new_pop, 'dimensions': population['dimensions']}


def ga_step(population, function_name, mutation_chance, crossover_chance, hill_climbing_chance, use_hill_climbing, max_eval):
    bit_mutation_chance = 1 - (1 - mutation_chance) ** (1/sum(population['dimensions']))
    population['values'] = crossover_population(population['values'], crossover_chance)
    population['values'] = mutate_population(population['values'], bit_mutation_chance)
    if use_hill_climbing:
        population['values'] = hybridise_hill_climbing(population, function_name, hill_climbing_chance, max_eval)
    results = evaluate_population(population, function_name)
    best_chromosome = min(list(zip(population['values'], results)), key=lambda x: x[1])
    # print(get_normal_values({'values': best_chromosome[0], 'dimensions': population['dimensions']}, function_name), best_chromosome[1])
    # print(best_chromosome[1])
    fitness = list(calculate_fitness(results))
    return generation_selection(population, fitness), best_chromosome[1]


def genetic_algorithm(function_name, max_eval, pop_size=100, generations=100, use_hill_climbing=True):
    global medii
    pop = initialize_population(function_name, pop_size)
    best_val = None
    # for _ in range(generations):
    #     pop = ga_step(pop, function_name, 0.3, 0.7, 0.05, use_hill_climbing, max_eval)
    while eval_calls <= max_eval:
        pop, pop_best = ga_step(pop, function_name, 0.3, 0.7, 0.05, use_hill_climbing, max_eval)
        best_val = best_val if best_val and best_val < pop_best else pop_best
        if use_hill_climbing:
            medii['gah'][eval_calls] = (medii['gah'][eval_calls] + best_val) / 2
        else:
            medii['ga'][eval_calls] = (medii['ga'][eval_calls] + best_val) / 2
    # print(best_val)


def only_hill_climbing(iterations):
    for restart in range(iterations):
        # name = 'rastrigin'
        # name = 'griewangk'
        # name = 'rosenbrock'
        name = 'six_hump'
        print('First improvement')
        val = hill_climbing(name, iterations=100)
        print(val)
        print(get_normal_values(val[0], name), val[1])
        print('------------------------------------------------------------------')
        print('Best improvement')
        val = hill_climbing(name, iterations=1000, best_improvement=True)
        print(val)
        print(get_normal_values(val[0], name), val[1])


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
    global medii
    global_best = None
    particle_best = [None]*pop_size
    x, v = pso_init_population(pop_size, vmax, function_name)
    iterations = 0
    while current_eval < max_eval:
        iterations += 1
        results = pso_evaluate_population(x, function_name)
        current_best = (results.index(min(results)), min(results))
        global_best = global_best if global_best and global_best[1] < current_best[1] else (x[current_best[0]], current_best[1])
        medii['pso'][current_eval] = (medii['pso'][current_eval] + global_best[1]) / 2
        for i in range(len(results)):
            particle_best[i] = particle_best[i] if particle_best[i] and particle_best[i][1] < results[i] else (x[i], results[i])
        x, v = pso_update_population(x, v, vmax, global_best[0], list(zip(*particle_best))[0], w, function_name)
        w[0] = w[0] - 0.9/max_iterations
    # print(global_best[1])


def call_ga(use_hillclimbing):
    # name = 'rastrigin'
    # name = 'griewangk'
    # name = 'rosenbrock'
    name = 'six_hump'
    genetic_algorithm(name, max_eval=50000, pop_size=100, generations=500, use_hill_climbing=use_hillclimbing)


def call_pso():
    # current_function = 'rastrigin'
    # current_function = 'griewangk'
    # current_function = 'rosenbrock'
    current_function = 'six_hump'
    hyper_w = [0.9, 2, 2]
    hyper_vmax = list(zip(*function_domains[current_function]))[1]
    pso(current_function, hyper_w, hyper_vmax, 20, 1000, max_eval=50000)


# eval_calls = 0

# only_hill_climbing(1)
# call_ga()
medii = {
    'ga': [0]*51000,
    'gah': [0]*51000,
    'pso': [0]*51000
}
for _ in range(30):
    eval_calls = 0
    call_ga(False)
print('finish ga')
for _ in range(30):
    eval_calls = 0
    call_ga(True)
print('finish gah')
for _ in range(30):
    current_eval = 0
    call_pso()
print('finish pso')

new_gah_values = [0]*51000

for i in range(509):
    gah_val_initiale = medii['gah']
    val_diferite = list(filter(lambda x: x != 0, gah_val_initiale[i*100+1:(i+1)*100]))
    if len(val_diferite) > 0:
        new_gah_values[(i + 1) * 100] = reduce(lambda acc, x: acc + x, val_diferite, 0) / len(val_diferite)
medii['gah'] = new_gah_values


ga_vals = list(zip(*filter(lambda x: x[1] != 0, enumerate(medii['ga']))))
gah_vals = list(zip(*filter(lambda x: x[1] != 0, enumerate(medii['gah']))))
pso_vals = list(zip(*filter(lambda x: x[1] != 0, enumerate(medii['pso']))))

ga_plot = plt.plot(ga_vals[0], ga_vals[1], label='GA')
gah_plot = plt.plot(gah_vals[0], gah_vals[1], label='GAH')
pso_plot = plt.plot(pso_vals[0], pso_vals[1], label='PSO')
plt.legend()
plt.savefig('six_hump10.png')
