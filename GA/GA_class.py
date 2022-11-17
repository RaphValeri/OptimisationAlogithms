import numpy as np
import matplotlib.pyplot as plt
from Fitness import Fitness
from Selection import Selection
from Crossover import Crossover
from Mutation import Mutation
from scipy.linalg import sqrtm
import copy

def elitism(prev_gen, new_gen, fitness_func, number = 1, option = "exclude" ):
    """
    process to select elites
    :param prev_gen: previous generation
    :param new_gen: new generation
    :param fitness_func: fitness function
    :param number: number of elites
    :param option: elitism method 'add', 'substitute' or else (smart substitution)
    :return: extracted elites
    """
    fit_eval = fitness_func(prev_gen)
    sorted_prev = prev_gen[:, (fit_eval).argsort()]
    elites = sorted_prev[:,:(number+1)]
    # simply add best from last gen to new gen. Eventually clones the best parents if already in new gen (selected and not mutated case)
    if option == "add":
        new_gen = np.hstack((new_gen, elites))
    #substituting weakest by best from previous gen. Eventually clones the best parents if already in pop
    elif option == "substitute":
        new_gen[:, fitness_func(new_gen).argsort()[-(number + 1):]] = elites
    # substituting elites to weaker individuals only if not already in pop
    else :
        #verifying if elites are already in new_gen. In not, we replace the weakest individuals by them
        bool_contains = np.array([elites.T.tolist()[i] in new_gen.T.tolist() for i in range(number)]).astype('int')
        if np.sum(bool_contains)==0:
            new_gen[:, fitness_func(new_gen).argsort()[-(number + 1):]] = elites
    return new_gen

def culling(A, fitness_func, number = 1):
    pop = copy.deepcopy(A)
    fit_eval = fitness_func(pop)
    pop = pop[:, fit_eval.argsort()]
    if number == 0:
        return pop
    new_pop = pop[:, :(-number)]
    return new_pop

class GA:
    def __init__(self, nb_genes, initial_pop_size, bound_min, bound_max, fitness_name, elites = 0, cull = 0):
        self.nb_genes = nb_genes
        self.initial_pop_size = initial_pop_size
        self.bound_min = bound_min
        self.bound_max = bound_max
        self.elites = elites
        self.cull = cull
        self.fitness = Fitness(fitness_name)
        self.selection = Selection(100, name = "naive", fitness_func = self.fitness.evaluate)
        self.select_number = 100
        self.crossover = None
        self.mutation = None

    def selection_init(self, number, name = "wheel", tournament_size = None, proba = None):
        self.selection = Selection(number, name = name, fitness_func = self.fitness.evaluate, tournament_size = tournament_size, proba=proba)
        self.select_number = number

    def crossover_init(self, child_nb, name = "uniform", alpha = 0, rate = 1):
        self.crossover = Crossover(child_nb = child_nb, name = name, alpha=alpha, rate = rate)

    def mutation_init(self, name = "shrink", width = 0, rate = 0):
        self.mutation = Mutation(name, width, rate)
        if width == 0 and name == "shrink":
            print("width = 0 -> Gaussian std is equal to max |pop|")

    def simulation(self,  nb_gen = 500, precision = 4, printer = True):
        parents = np.random.uniform(self.bound_min, self.bound_max, (self.nb_genes, self.initial_pop_size))
        if printer:
            print("Gen 0 best minimization = {}".format(round(self.fitness.evaluate(parents).min(), precision)))
        if self.mutation.name == "cmaes":
            for i in range(1, nb_gen+1):
                selected_parents, idx = self.selection.apply(parents)
                # cmaes mutation
                children = self.mutation.apply(parents, idx=idx)
                if i % 10 == 0 and printer:
                    print("Gen {} best minimization = {}".format(i, round(self.fitness.evaluate(children).min(), precision)))
                parents = children
        else:
            all_width = np.linspace(1e-4, self.mutation.initial_std_width, nb_gen)[::-1]
            all_scores = []
            for i in range(1, nb_gen+1):
                selected_parents, idx = self.selection.apply(parents)
                #selected_parents = elitism(parents, selected_parents, self.fitness.evaluate, number=1)
                # all crossed children
                children = self.crossover.apply(selected_parents)
                # mutation
                children = self.mutation.apply(children, width = all_width[i-1])
                # elitism & culling
                children = elitism(selected_parents, children, self.fitness.evaluate, number=self.elites)
                children = culling(children, self.fitness.evaluate, number = self.cull)
                if i % 10 == 0 and printer:
                    #print("---------------------------------------------")
                    score = round(self.fitness.evaluate(children).min(), precision)
                    all_scores.append(score)
                    print("Gen {} best minimization = {}".format(i, score))
                parents = children
                # if len(all_scores)>3 and abs(all_scores[-1] - all_scores[-3])<(10**(-precision)) and i<nb_gen-1:
                #     all_width[i+1] *= 5
        return children, children[:, np.argmin(self.fitness.evaluate(children))]

    def multiple_runs(self, nb_runs=10, nb_gen=1000, precision=4,printer=False):
        minimal_val = np.zeros(nb_runs)
        results = np.zeros((self.nb_genes, nb_runs))
        for i in range(nb_runs):
            children, best_child = self.simulation(nb_gen = nb_gen, precision = precision, printer = printer)
            fit_eval = np.around(self.fitness.evaluate(best_child[:,None]),precision)
            minimal_val[i] = fit_eval
            results[:, i] = best_child
            print(f"Run {i} best fitness = {fit_eval}")
            self.mutation.reinitialize_maes()
        avg_val = round(np.mean(minimal_val), precision)
        std_val = round(np.std(minimal_val), precision)
        best_overall = np.around(results[:, np.argmin(self.fitness.evaluate(results))], precision)
        print("======================================================")
        print(f"{nb_runs} simulations of {nb_gen} generations report:")
        print("Average fitness = ", avg_val)
        print("Average fitness Std = ", std_val)
        print("Best individual overall = ", np.around(best_overall, precision))
        return best_overall, results, avg_val, std_val


if __name__ == "__main__":
    experience = GA(nb_genes=10, initial_pop_size=1000, bound_min=-20, bound_max=20, fitness_name="rosenbrock", elites = 25, cull = 50)
    experience.selection_init(number=90, name="wheel", tournament_size =10, proba = 0.9)
    experience.crossover_init(child_nb = 1000, name="uniform", alpha=0.7, rate = 0.4)
    experience.mutation_init(name="shrink", width = 0, rate=0.15)
    #best, results, avg, std = experience.multiple_runs(nb_runs=3, nb_gen=500, precision=4, printer=1)

    exp2 = GA(nb_genes=10, initial_pop_size=2000, bound_min=-100, bound_max=100, fitness_name="rosenbrock", elites = 0, cull = 0)
    exp2.selection_init(number=100, name="naive", tournament_size=10, proba=1)
    exp2.mutation_init(name="cmaes")
    best, results, avg, std = exp2.multiple_runs(nb_runs=3, nb_gen=250, precision=4, printer=1)

