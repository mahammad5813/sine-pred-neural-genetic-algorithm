import numpy as np
from concurrent.futures import ProcessPoolExecutor
from nn import NN
from copy import copy, deepcopy
from nn_funcs import select_parent, crossover
import math
from plot_funcs import plot_the_best, plot_func, plot_bunch_nn



def test_func(inputs: tuple) -> int:
    x = inputs[0]

    return math.sin(x)

def fitness(true, pred):
    return sum(1/(1+abs(true[i]-pred[i])) for i in range(len(true)))
    

def normalize(value, min, max):
    return (value - min)/(max-min)

def evaluate_fitness(nn: NN, dataset):
    score = fitness(dataset[:, 1], nn.evaluate_bunch(dataset[:,0].reshape(-1,1)))
    return score

def main():
    data_min = 0
    data_max = 6

    plot_min, plot_max = 0,6

    n_population: int = 2000
    n_gens: int = 100

    nn_shape: tuple = (1,20,10,10,10,1)
    population: list[NN] = [NN(nn_shape) for _ in range(n_population)]

    executor = ProcessPoolExecutor(max_workers=4)

    gen_bests = []

    for gen_i in range(n_gens):

        dataset = np.random.uniform(data_min,data_max,(100,2))

        dataset[:, 1] = np.apply_along_axis(test_func, 1, dataset[:,0].reshape(-1,1))
        
        fitnesses = list(executor.map(evaluate_fitness, population, [dataset]*len(population)))

        for i, nn in enumerate(population):
            nn.score = fitnesses[i]


        population.sort(key=lambda x: x.score, reverse=True)
        
        population = population[:int(n_population/2)]
        
        print('Gen:',gen_i, population[0].evaluate([3]), test_func([3]), population[0].score)

        gen_bests.append(population[0])

        
        min_score = min([nn.score for nn in population])
        max_score = max([nn.score for nn in population])

        for i, nn in enumerate(population):
            nn.score = (nn.score-min_score)/(max_score-min_score)

        bests = population[:10]

        if gen_i>=n_gens-1:
            break

        temp_population = copy(population)

        population = []

        for _ in range(int((n_population-10)/2)):
            genes = crossover(select_parent(temp_population), select_parent(temp_population))

            population.append(NN(nn_shape, genes[0][0], genes[0][1]))
            population.append(NN(nn_shape, genes[1][0], genes[1][1]))

        for best in bests:
            population.append(deepcopy(best))

        for nn in population:
            nn.mutate()
            nn.score = 0


    plot_the_best(bests[0],plot_min, plot_max)

    plot_bunch_nn(gen_bests, min=plot_min, max=plot_max)

    plot_func(test_func, plot_min, plot_max)

if __name__=="__main__":

    main()

