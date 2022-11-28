import numpy as np
#from Fitness import Fitness
from abc import ABC, abstractmethod

def uniform_crossover(x,y, alpha = 0.5, crossover_rate = 1):
    """
    uniform crossing
    :param x: parent x (numpy : shape = (n,1))
    :param y: parent y (numpy : shape = (n,1))
    :param alpha: hyperparameter
    :return: crossover children
    """
    crossover_event = (np.random.uniform(0,1)<crossover_rate)
    if crossover_event:
        nb_points = x.shape[0]
        choices = np.array([0, 1])
        weights = np.array([1-alpha, alpha])
        random_factor = np.random.choice(choices, size=(nb_points, 1), p=weights)
        child1 = random_factor*x + (1-random_factor)*y
        child2 = random_factor*y + (1-random_factor)*x
    else:
        child1 = x
        child2 = y
    return np.hstack((child1, child2))

def k_points_crossover(x,y, alpha = 2, crossover_rate = 1):
    """
    k_points crossing
    :param x: parent x (numpy : shape = (n,1))
    :param y: parent y (numpy : shape = (n,1))
    :param alpha: hyperparameter
    :return: crossover children
    """
    crossover_event = (np.random.uniform(0, 1) < crossover_rate)
    if crossover_event:
        nb_points = x.shape[0]
        k_points = np.random.choice(nb_points-1, size = alpha, replace = False)
        k_points = np.sort(k_points)
        genes_fact = np.zeros((nb_points,1))
        if alpha==1:
            genes_fact[k_points[0]:]=1
            child1 = genes_fact * x + (1 - genes_fact) * y
            child2 = genes_fact * y + (1 - genes_fact) * x
            return np.hstack((child1, child2))
        elif alpha%2 == 0:
            for i in range(alpha):
                if i%2==0:
                    genes_fact[k_points[i]+1:k_points[i+1]+1] = 1
                if (i+1)==alpha-1:
                    break
        else:
            for i in range(alpha):
                if i==alpha-1:
                    genes_fact[k_points[i]+1:] = 1
                    break
                if i%2==0:
                    genes_fact[k_points[i]+1:k_points[i+1]+1] = 1
        child1 = genes_fact*x + (1-genes_fact)*y
        child2 = genes_fact*y + (1-genes_fact)*x
    else :
        child1 = x
        child2 = y
    return np.hstack((child1, child2))

def blx_alpha_crossover(x,y, alpha = 0.1, crossover_rate=1):
    """
    crossover genetic operation by blx_alpha
    :param x: parent x (numpy : shape = (n,1))
    :param y: parent y (numpy : shape = (n,1))
    :param alpha: hyperparameter
    :return: crossover child
    """
    i_size = len(x)
    crossover_event = (np.random.uniform(0, 1) < crossover_rate)
    if crossover_event:
        nx, ny = x.reshape((i_size,1)), y.reshape((i_size, 1))
        conc_xy = np.hstack((nx,ny))
        min_conc = conc_xy.min(axis=1)
        max_conc = conc_xy.max(axis=1)
        dist = np.abs(max_conc - min_conc)
        child = np.random.uniform(min_conc - alpha*dist, max_conc + alpha*dist)
        return child.reshape((i_size, 1))
    else:
        r = np.random.choice([0,1], p = [0.5, 0.5])
        child = r*x + (1-r)*y
        return child.reshape((i_size, 1))

def sbx_crossover(x,y, alpha = 5, crossover_rate=1):
    """
    crossover genetic operation by blx_alpha
    :param x: parent x (numpy : shape = (n,1))
    :param y: parent y (numpy : shape = (n,1))
    :param alpha: hyperparameter
    :return: crossover child
    """
    i_size = len(x)
    crossover_event = (np.random.uniform(0, 1) < crossover_rate)
    if crossover_event:
        u = np.random.uniform(0,1)
        beta = (2*u)**(1/(alpha+1)) if u <= 0.5 else (1/(2*(1-u)))**(1/(alpha+1))
        child1 = 0.5*((1+beta)*x + (1-beta)*y)
        child2 = 0.5*((1-beta)*x + (1+beta)*y)
    else:
        child1 = x
        child2 = y
    return np.hstack((child1, child2))

def crossing(S, fitness, crossover = blx_alpha_crossover, child_nb=100, alpha=0.1, crossover_rate =1):
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
        children = crossover(par1,par2, alpha=alpha, crossover_rate= crossover_rate)
        A = np.hstack((A, child))
    A = A[:,1:]
    cross_score = fitness(A)
    sorted_A = A[:, (cross_score).argsort()]
    return sorted_A

class Crossover:
    def __init__(self, child_nb, name="uniform", alpha = 0.1, rate = 1):
        self.child_nb = child_nb
        self.possible_crossovers = {"uniform": uniform_crossover,
                                    "k-points": k_points_crossover,
                                   "blx-alpha":blx_alpha_crossover,
                                    "sbx":sbx_crossover}
        if name not in self.possible_crossovers.keys():
            print("Warning, crossover name '{}' is not recognized".format(name))
            print("Admissible crossover names are : 'uniform', 'blx-alpha', 'k-points', 'sbx'")
            print("uniform is taken by default")
            self.name = "uniform"
        else:
            self.name = name
        self.function =  self.possible_crossovers[self.name]
        self.alpha = alpha
        self.rate = rate

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
            child = self.function(par1, par2, alpha=self.alpha, crossover_rate = self.rate)
            children  = np.hstack((children , child))
        children  = children [:, 1:]
        # cross_score = fitness_func(children)
        # sorted_children  = children[:, (cross_score).argsort()]
        return children


    def __str__(self):
        return "{} crossover".format(self.name)


