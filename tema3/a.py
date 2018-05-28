import random
import copy
import numpy as np


def read_graph(filename):
    with open(filename, 'r') as f:
        _, _, v, e = f.readline().split()
        graph = [[] for _ in range(int(v))]
        for line in f.readlines():
            _, x, y = line.split()
            x = int(x) - 1
            y = int(y) - 1
            graph[x] += [y]
            graph[y] += [x]
    return graph


def ff(graph, node_order):
    color_number = 1
    colors = {1}
    assigned_colors = [None] * len(node_order)
    for node in node_order:
        cant_use = set()
        for neighbour in graph[node]:
            if assigned_colors[neighbour]:
                cant_use.add(assigned_colors[neighbour])
        possible_colors = list(colors - cant_use)
        if len(possible_colors) is 0:
            color_number += 1
            colors.add(color_number)
            possible_colors.append(color_number)
        assigned_colors[node] = possible_colors[0]
    return color_number


def pso_initialize(pop_size, instance_size):
    population = []
    for _ in range(pop_size):
        population.append({'location': np.random.rand(instance_size), 'velocity': np.random.rand(instance_size)})
    return population


def test_particle(particle, graph):
    sorted_list = sorted(enumerate(particle['location']), key=lambda x: x[1])
    node_order = list(map(lambda x: x[0], sorted_list))
    return ff(graph, node_order)


def fitness_particle(particle, graph):
    return 1/test_particle(particle, graph)


def update_particle(particle, p_best, g_best, c1, c2):
    new_particle = {'location': None, 'velocity': None}
    inertia = particle['velocity']
    cognitive = c1 * random.random() * (p_best - particle['location'])
    social = c2 * random.random() * (g_best - particle['location'])
    new_particle['velocity'] = inertia + cognitive + social
    new_particle['location'] = particle['location'] + new_particle['velocity']
    return new_particle


def pso(graph, pop_size, c1, c2, iterations):
    p_best = [{'fitness': 0, 'location': None} for _ in range(pop_size)]
    g_best = {'fitness': 0, 'location': None}
    population = pso_initialize(pop_size, len(graph))
    for _ in range(iterations):
        for index, particle in enumerate(population):
            fitness = fitness_particle(particle, graph)
            if fitness > p_best[index]['fitness']:
                p_best[index]['fitness'] = fitness
                p_best[index]['location'] = copy.deepcopy(particle['location'])
            if fitness > g_best['fitness']:
                g_best['fitness'] = fitness
                g_best['location'] = copy.deepcopy(particle['location'])
        for index, particle in enumerate(population):
            population[index] = update_particle(particle, p_best[index]['location'], g_best['location'], c1, c2)
    print(g_best)
    print(test_particle(g_best, graph))


# current_graph = read_graph('instante/myciel7.col')
# current_graph = read_graph('instante/miles1500.col')
current_graph = read_graph('instante/fpsol2.i.3.col')
pso(current_graph, 20, 2, 2, 1000)
