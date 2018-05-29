import random

import numpy
from scipy.optimize import linear_sum_assignment

nr_max_eval =50

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

def initializare(nr_lin, nr_col, k):
    return [[random.randint(0, k-1) for j in range(nr_col)] for i in range(nr_lin)]

def aliniere(colorare1, colorare2, k):
    cost = [[[0] for j in range(k)] for i in range(k)]
    for c1 in range(k):
        for c2 in range(k):
            indici1 = set([i for i, x in enumerate(colorare1) if x == c1])
            indici2 = set([i for i, x in enumerate(colorare2) if x == c2])
            intersectie = indici1 & indici2
            cost[c1][c2] = len(intersectie) #len(indici1) + len(indici2) - len(intersectie) #sau len(colorare1)-len ???
    (row_ind,col_ind) = linear_sum_assignment(cost)
    colorare_aliniata_2 = [-1 for i in range(len(colorare1))]
    for index in range(len(row_ind)):
        color_to_replace = col_ind[index]
        indices_to_change = [i for i, x in enumerate(colorare2) if x == color_to_replace]
        for index_tc in indices_to_change:
            colorare_aliniata_2[index_tc] = colorare1[row_ind[index]]
    if colorare_aliniata_2.count(-1)>0:
        print("ceva aiurea in aliniere")
    return (colorare1, colorare_aliniata_2)

def incrucisare_parinti(parinti):
    copil = [parinti[random.randint(0,1)][i] for i in range(len(parinti[0]))]
    return copil

def vertex_descent(colorare, graf, k):
    n = len(colorare)
    cost_nod_culoare = [[0 for j in range(k)] for i in range(n)]
    best_value = []
    best_colrs = []
    for nod in range(n):
        for vec in graf[nod]:
            cost_nod_culoare[nod][colorare[vec]] += 1
        best_value += [min(cost_nod_culoare[nod])]
        best_colrs += [set([i for i, x in enumerate(cost_nod_culoare[nod]) if x == best_value[nod]])]
    nr_iter_fara_sch = 10
    iter_fara_sch = nr_iter_fara_sch
    while iter_fara_sch>0:
        iter_fara_sch -= 1
        for nod in range(n):
            clr = random.sample(best_colrs[nod],1)[0]
            if (cost_nod_culoare[nod][clr]<cost_nod_culoare[nod][colorare[nod]]):
                iter_fara_sch = nr_iter_fara_sch
            for vec in graf[nod]:
                cost_nod_culoare[vec][colorare[nod]] -= 1
                if (cost_nod_culoare[vec][colorare[nod]]<best_value[vec]):
                   best_value[vec] = cost_nod_culoare[vec][colorare[nod]]
                   best_colrs[vec] = set()
                if (cost_nod_culoare[vec][colorare[nod]] == best_value[vec]):
                   best_colrs[vec].add(colorare[nod])
                cost_nod_culoare[vec][clr] += 1
                if (cost_nod_culoare[vec][clr]-1 == best_value[vec]):
                    best_value[vec] = min(cost_nod_culoare[vec])
                    best_colrs[vec] = set([i for i, x in enumerate(cost_nod_culoare[vec]) if x == best_value[vec]])
            colorare[nod]=clr
    return colorare


def incrucisare(populatie, nr_incrucisari, k):
    for incrucisare in range(nr_incrucisari):
        parinte1 = populatie[random.randint(0, len(populatie)-1)]
        parinte2 = populatie[random.randint(0, len(populatie)-1)]
        #populatie.remove(parinte1)
        #populatie.remove(parinte2)
        #print("aliniere")
        (parinte1,parinte2) = aliniere(parinte1,parinte2, k)
        #print("aliniere")
        #print("inc_par")
        copil = incrucisare_parinti([parinte1,parinte2])
        #print("inc_par")
        #print("ver_des")
        copil = vertex_descent(copil, graf, k)
        #print("ver_des")
        populatie.append(copil)
    return populatie

def fitness(cromozom, graf):
    conflicte = 0
    for i in range(len(graf)):
        for j in range(len(graf[i])):
            if (cromozom[i]==cromozom[graf[i][j]]):
                conflicte+=1
    return len(graf)**2 - conflicte/2

def selectie(populatie, pop_size, graf):
    populatie = sorted(populatie, key = lambda cromozom: fitness(cromozom,graf), reverse = True)[:pop_size]
    return populatie

def cauta_solutie(graf, k):
    n=len(graf)
    pop_size = 40
    pc = 0.7
    #pm = 1 - (0.7 ** (1 / dim_reprez))
    populatie = initializare(pop_size,n,k)
    nr_eval_fct = 0
    while nr_eval_fct <= nr_max_eval:
        l1=len(populatie)
        best = max(populatie, key=lambda cromozom: fitness(cromozom, graf))
        #print(n * n - fitness(best, graf))
        if (fitness(best, graf) == n * n):
            break
        #print(1)
        populatie = incrucisare(populatie, pop_size//5, k)
        #print(2)
        populatie = selectie(populatie, pop_size, graf)
        #print(3)
        print(nr_eval_fct)
        nr_eval_fct +=1
        print(n * n - fitness(best, graf))

    print(best)

graf = read_graph('miles1500.col')
cauta_solutie(graf,72)