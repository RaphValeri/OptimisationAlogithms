import numpy as np
#from Fitness import Fitness
import copy

def wheel(A, fitness):
    """
    wheel roulette by cumsum
    :param A: All individuals
    :param fitness: fitness function
    :param func: evaluation function
    :return: Selected individual according to roulette selection
    """
    nb_indiv = A.shape[1]
    fit_eval = fitness(A)
    # making proba based on cumulative fitness
    inv_fit = 1/(abs(fit_eval)+ 1e-8)
    inv_fit /= np.sum(inv_fit)
    cumul_proba = np.cumsum(inv_fit)
    cumul_proba[-1] = 1
    return cumul_proba

def wheel_selection(A, iter, fitness):
    """
    selecting fitted individuals by wheel roulette
    :param A: all individuals
    :param iter: portion of individuals to select
    :param fitness: the fitness function
    :return: sorted array of selected individuals
    """
    #iter = A.shape[1]
    cumul_proba = wheel(A, fitness)
    # finding 'iter' number between 0 and 1
    r = np.random.uniform(0,1, iter)
    #extracting the corresponding indexes in cumul_proba
    select_idx = cumul_proba.searchsorted(r)
    # extracting population corresponding to the indexes
    selected_pop = A[:, select_idx]
    fit_eval = fitness(selected_pop)
    #return sorted array of individuals (descending fitness)
    return selected_pop[:, (fit_eval).argsort()], select_idx


def naive_selection(A, number, fitness):
    """
    selecting fitted individuals
    :param A: all individuals
    :param number: number of individuals to select
    :param fitness: the fitness function
    :return: sorted array of selected individuals
    """
    #number = A.shape[1]
    #evaluate fitness of population
    fit_eval = fitness(A)
    # sorting by best fitness
    sorted = A[:,(fit_eval).argsort()]
    # selecting
    select = sorted[:,:number]
    return select, fit_eval.argsort()[:number]

def random_selection(A,number, fitness = None):
    """
    selecting random individuals
    :param A: all individuals
    :param number: number of individuals to select (int)
    :return: selected individuals in 2D array
    """
    pop_size = A.shape[1]
    rnd_index = np.random.randint(0, pop_size, number)
    return A[:, rnd_index], rnd_index

def tournament_selection(A,number, tournament_size=3, fitness = None):
    """
    selecting individuals by tournament
    :param A: all individuals
    :param number: number of individuals to select (int)
    :return: selected individuals in 2D array
    """
    nb_genes = A.shape[0]
    pop_size = A.shape[1]
    pop = copy.deepcopy(A)
    selected = np.zeros((nb_genes, number))
    np.random.shuffle(pop.T)
    fit_eval = fitness(pop)
    winner_index = np.array([np.argwhere(fit_eval==np.min(fit_eval[i*tournament_size:(i+1)*tournament_size]))[0][0]
                             for i in range(number)])
    selected = pop[:, winner_index]
    selected_fit_eval = np.sort(fitness(selected))
    return selected, winner_index

def tournament_selection2(A,number, tournament_size=3, proba = 0.9, fitness = None):
    """
    selecting individuals by tournament
    :param A: all individuals
    :param number: number of individuals to select (int)
    :return: selected individuals in 2D array
    """
    nb_genes = A.shape[0]
    pop_size = A.shape[1]
    pop = copy.deepcopy(A)
    selected = np.zeros((nb_genes, number))
    winner_index = np.zeros(number)
    for i in range(number):
        index = np.random.choice(pop_size, size = tournament_size, replace = False)
        tournament = pop[:, index]
        fit_eval = np.around(fitness(tournament),4)
        dic_eval = {fit_eval[i] : index[i] for i in range(tournament_size)}
        tournament = tournament[:, fit_eval.argsort()]
        p_weights = np.array([proba*(1-proba)**i for i in range(tournament_size)])
        p_weights[-1] = 1 - np.sum(p_weights[:-1])
        selected_index = np.random.choice(tournament_size, p = p_weights)
        winner = tournament[:, selected_index]
        selected[:, i] = winner
        fit_winner = round(fitness(winner[:,None])[0], 4)
        winner_index[i]= int(dic_eval[fit_winner])
    return selected, winner_index


class Selection:
    """
    Selecton class
    """
    def __init__(self, number, name="wheel", fitness_func=None,tournament_size = None, proba = None):
        """

        :param number:
        :param name: 'wheel', 'naive', 'tournament', 'random'
        :param fitness_func: the fitness function
        :param tournament_size: int
        :param proba: winner proba to be selected in tournament selection
        """
        self.possible_selections = {"random" : random_selection,
                                   "naive":naive_selection,
                                   "roulette":wheel_selection,
                                   "tournament":tournament_selection2}
        if name not in self.possible_selections.keys():
            print("Warning, selection name is not recognized")
            print("Admissible selection names are : 'random', 'naive', 'roulette', 'tournament'")
            print("wheel selection is taken by default")
            self.name = "roulette"
        else :
            self.name = name
        self.number = number
        self.fitness_func = fitness_func
        self.tournament_size = tournament_size
        self.proba = proba
        if self.name == "tournament":
            self.function = lambda A,number , fitness : tournament_selection2(A,
                                                                           number,
                                                                           tournament_size=self.tournament_size,
                                                                           proba = self.proba,
                                                                           fitness = fitness)
        else:
            self.function = self.possible_selections[self.name]

    def apply(self, population):
        """
        Apply selection
        :param population: array (n, ...)
        :return: Selected parents from population
        """
        return self.function(population, self.number, self.fitness_func)

    def __str__(self):
        return "{} selection".format(self.name)

