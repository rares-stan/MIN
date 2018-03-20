import math
from functools import reduce
import random
import copy

discretization = 100
function_dimensions = {
    'rastrigin': 30,
    'griewangk': 2,
    'rosenbrock': 3,
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


def hill_climbing(function_name, start_chromosome=None, iterations=1000, best_improvement=False):
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


def hybridise_hill_climbing(population, function_name, mutate_chance):
    return list(
        map(
            lambda chromosome:
                hill_climbing(
                    function_name,
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


def ga_step(population, function_name, mutation_chance, crossover_chance, hill_climbing_chance, use_hill_climbing):
    bit_mutation_chance = 1 - (1 - mutation_chance) ** (1/sum(population['dimensions']))
    population['values'] = crossover_population(population['values'], crossover_chance)
    population['values'] = mutate_population(population['values'], bit_mutation_chance)
    population['values'] = hybridise_hill_climbing(population, function_name, hill_climbing_chance) if use_hill_climbing else None
    results = evaluate_population(population, function_name)
    best_chromosome = min(list(zip(population['values'], results)), key=lambda x: x[1])
    # print(get_normal_values({'values': best_chromosome[0], 'dimensions': population['dimensions']}, function_name), best_chromosome[1])
    print(best_chromosome[1])
    fitness = list(calculate_fitness(results))
    return generation_selection(population, fitness)


def genetic_algorithm(function_name, pop_size=100, generations=100, use_hill_climbing=True):
    pop = initialize_population(function_name, pop_size)
    for _ in range(generations):
        pop = ga_step(pop, function_name, 0.3, 0.7, 0.05, use_hill_climbing)


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


def call_ga():
    name = 'rastrigin'
    # name = 'griewangk'
    # name = 'rosenbrock'
    # name = 'six_hump'
    genetic_algorithm(name, pop_size=500, generations=500, use_hill_climbing=True)


# only_hill_climbing(1)
call_ga()
