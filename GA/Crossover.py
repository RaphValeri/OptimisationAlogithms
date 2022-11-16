import numpy as np
from Fitness import Fitness
from abc import ABC, abstractmethod

def uniform_crossover(x,y, alpha = 0.5):
    """
    uniform crossing
    :param x: parent x (numpy : shape = (n,1))
    :param y: parent y (numpy : shape = (n,1))
    :param alpha: hyperparameter
    :return: crossover children
    """
    nb_points = x.shape[0]
    choices = np.array([0, 1])
    weights = np.array([1-alpha, alpha])
    random_factor = np.random.choice(choices, size=(nb_points, 1), p=weights)
    child1 = random_factor*x + (1-random_factor)*y
    child2 = random_factor*y + (1-random_factor)*x
    return np.hstack((child1, child2))

def k_points_crossover(x,y, alpha = 2):
    """
    k_points crossing
    :param x: parent x (numpy : shape = (n,1))
    :param y: parent y (numpy : shape = (n,1))
    :param alpha: hyperparameter
    :return: crossover children
    """
    nb_points = x.shape[0]
    k_points = np.random.choice(nb_points, size = alpha, replace = False)
    k_points = np.sort(k_points)
    genes_fact = np.zeros((nb_points,1))
    if alpha==1:
        genes_fact[k_points[0]:]=1
        child1 = genes_fact * x + (1 - genes_fact) * y
        child2 = genes_fact * y + (1 - genes_fact) * x
        return np.hstack((child1, child2))

    for i in range(0, alpha-1):
        if i%2==0:
            genes_fact[k_points[i]:k_points[i+1]] = 1
    child1 = genes_fact*x + (1-genes_fact)*y
    child2 = genes_fact*y + (1-genes_fact)*x
    return np.hstack((child1, child2))

def blx_alpha_crossover(x,y, alpha = 0.1):
    """
    crossover genetic operation by blx_alpha
    :param x: parent x (numpy : shape = (n,1))
    :param y: parent y (numpy : shape = (n,1))
    :param alpha: hyperparameter
    :return: crossover child
    """
    i_size = len(x)
    nx, ny = x.reshape((i_size,1)), y.reshape((i_size, 1))
    conc_xy = np.hstack((nx,ny))
    min_conc = conc_xy.min(axis=1)
    max_conc = conc_xy.max(axis=1)
    dist = np.abs(max_conc - min_conc)
    child = np.random.uniform(min_conc - alpha*dist, max_conc + alpha*dist)
    return child.reshape((i_size, 1))

def crossing(S, fitness, crossover = blx_alpha_crossover, child_nb=100, alpha=0.1):
    """
    Effective crossing of selected population
    :param S: selected population by (descending) sorted fitness
    :param child_nb:
    :param alpha:
    :return:
    """
    nb_points = S.shape[0]
    pop = S.shape[1]
    A = np.zeros((nb_points,1))
    while A.shape[1]<child_nb+1:
        i = np.random.randint(pop)
        j = np.random.randint(pop)
        while i==j:
            j = np.random.randint(pop)
        par1 = S[:,i]
        par2 = S[:,j]
        children = crossover(par1,par2, alpha=alpha)
        A = np.hstack((A, child))
    A = A[:,1:]
    #adding best parents
    #A = np.hstack((A, S[:,:(elites + 1)]))
    cross_score = fitness(A)
    sorted_A = A[:, (cross_score).argsort()]
    return sorted_A

class Crossover:
    def __init__(self, child_nb, name="uniform", alpha = 0.1):
        self.child_nb = child_nb
        self.possible_crossovers = {"uniform": uniform_crossover,
                                    "k_points": k_points_crossover,
                                   "blx_alpha":blx_alpha_crossover}
        if name not in self.possible_crossovers.keys():
            print("Warning, crossover name '{}' is not recognized".format(name))
            print("Admissible crossover names are : 'uniform', 'blx_alpha'")
            print("uniform is taken by default")
            self.name = "uniform"
        else:
            self.name = name
        self.function =  self.possible_crossovers[self.name]
        self.alpha = alpha

    def apply(self, population, k=10):
        """
        Effective crossing of selected population
        :param S: selected population by (descending) sorted fitness
        :param child_nb:
        :param alpha:
        :return:
        """
        nb_points = population.shape[0]
        pop_size= population.shape[1]
        children = np.zeros((nb_points, 1))
        while children.shape[1] < self.child_nb + 1:
            i = np.random.randint(pop_size)
            j = np.random.randint(pop_size)
            while i == j:
                j = np.random.randint(pop_size)
            par1 = population[:, i][:,None]
            par2 = population[:, j][:,None]
            child = self.function(par1, par2, alpha=self.alpha)
            children  = np.hstack((children , child))
        children  = children [:, 1:]
        # cross_score = fitness_func(children)
        # sorted_children  = children[:, (cross_score).argsort()]
        return children


    def __str__(self):
        return "{} crossover".format(self.name)


