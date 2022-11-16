import numpy as np
import matplotlib.pyplot as plt
from optproblems import cec2005


"""
What is needed to solve GA problem :
- population of individuals
- fitness function
- genetic operations (ex. mutation & crossover)
(- indivuals selection for crossover)
"""

def rastrigin(x):
    """
    Rastragin benchmark function
    :param x: 1D array (numpy : shape = (n,1))
    :return: Rastragin function evaluation in x
    """
    A = 10
    n = len(x)
    S = np.sum(x**2 - A*np.cos(2*np.pi*x), axis = 0)
    return A*n + S

def rosenbrock(x):
    """
    Rosenbrock benchmark function
    :param x: 1D array of size >2 (numpy : shape = (n,1))
    :return: Rosenbrock function evaluation in x
    """
    n = len(x)
    if n<2:
        print("Array length must be > 2")
        return False
    if len(x.shape)<2:
        x = x.reshape((n, 1))
    xi = x[:-1, :]
    xp1 = x[1:, :]
    S = np.sum(100*(xp1 - xi**2)**2 + (1-xi)**2, axis = 0)
    return S

def sphere(x):
    """
    Sphere function
    :param x: 1D array of size >2 (numpy : shape = (n,1))
    :return: sphere evaluation
    """
    S = np.sum(x**2, axis = 0)
    return S

def booth(x):
    S = (x[0] + 2*x[-1]-7)**2 + (2*x[0] + x[-1] -5)**2
    return S

def wheel(A, fitness):
    """
    wheel roulette by cumsum
    :param A: All individuals
    :param fitness: fitness function
    :param func: evaluation function
    :return: Selected individual according to roulette
    """
    nb_indiv = A.shape[1]
    fit_eval = fitness(A)
    inv_fit = 1/(abs(fit_eval)+ 1e-8)
    inv_fit /= np.sum(inv_fit)
    cumul_proba = np.cumsum(inv_fit)
    cumul_proba[-1] = 1
    return cumul_proba

def wheel_selection(A, iter, fitness):
    """
    selecting fitted individuals by wheel roulette
    :param A: all individuals
    :param iter: number of individuals to select (int)
    :param fitness: the fitness function
    :return: sorted array of selected individuals
    """
    cumul_proba = wheel(A, fitness)
    # finding 'iter' number between 0 and 1
    r = np.random.uniform(0,1, iter)
    #extracting the corresponding indexes in cumul_proba
    select_idx = cumul_proba.searchsorted(r)
    # extracting population corresponding to the indexes
    selected_pop = A[:, select_idx]
    fit_eval = fitness(selected_pop)
    #return sorted array of individuals (descending fitness)
    return selected_pop[:, (fit_eval).argsort()]


def naive_selection(A, number, fitness):
    """
    selecting fitted individuals
    :param A: all individuals
    :param number: number of individuals to select (int)
    :param fitness: the fitness function
    :return: sorted array of selected individuals
    """
    #evaluate fitness of population
    fit_eval = fitness(A)
    # sorting by best fitness
    sorted = A[:,(fit_eval).argsort()]
    # selecting
    select = sorted[:,:number]
    return select

def random_selection(A, number):
    """
    selecting random individuals
    :param A: all individuals
    :param number: number of individuals to select (int)
    :return:
    """
    pop_size = A.shape[1]
    rnd_index = np.random.randint(0, pop_size, number)
    return A[:, rnd_index]


def uniform_crossover(x,y, alpha = 0.5):
    """
    uniform crossing
    :param x: parent x (numpy : shape = (n,1))
    :param y: parent y (numpy : shape = (n,1))
    :param alpha: hyperparameter
    :return: crossover child
    """
    nb_points = x.shape[0]
    choices = np.array([0, 1])
    weights = np.array([alpha, 1-alpha])
    random_factor = np.random.choice(choices, size=(nb_points, 1), p=weights)
    child = random_factor*x + (1-random_factor)*y
    return child

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
        child = crossover(par1,par2, alpha=alpha)
        A = np.hstack((A, child))
    A = A[:,1:]
    #adding best parents
    #A = np.hstack((A, S[:,:(elites + 1)]))
    cross_score = fitness(A)
    sorted_A = A[:, (cross_score).argsort()]
    return sorted_A

def elitism(prev_gen, new_gen, fitness, number = 2, option = "add" ):
    """
    process to select elites
    :param pop: population
    :param number: number of elites
    :return: extracted elites
    """
    fit_eval = fitness(prev_gen)
    sorted_prev = prev_gen[:, (fit_eval).argsort()]
    elites = sorted_prev[:,:(number+1)]
    # simply add best from last gen to new gen. Eventually clones the best parents if already in new gen (selected and not mutated case)
    if option == "add":
        new_gen = np.hstack((new_gen, elites))
    #substituting weakest by best from previous gen. Eventually clones the best parents if already in pop
    elif option == "substitute":
        new_gen[:, fitness(new_gen).argsort()[-(number + 1):]] = elites
    # substituting elites to weaker individuals only if not already in pop
    else :
        #verifying if elites are already in new_gen. In not, we replace the weakest individuals by them
        bool_contains = np.array([elites.T.tolist()[i] in new_gen.T.tolist() for i in range(number)]).astype('int')
        if np.sum(bool_contains)==0:
            new_gen[:, fitness(new_gen).argsort()[-(number + 1):]] = elites
    return new_gen

def shrink_mutation(X, mutation_rate = 0):
    """
    muation of X by gaussian shrink
    :param X: indiv / pop
    :return:s shrinked mutated X
    """
    nb_points = X.shape[0]
    nb_indiv = X.shape[1]
    choices = np.array([0,1])
    weights = np.array([1-mutation_rate, mutation_rate ])
    bin_factor = np.random.choice(choices, size = X.shape, p = weights)
    adder = np.random.normal(loc = 0, scale = np.max(abs(X)), size=X.shape )
    mutated = X + bin_factor*adder
    return mutated


def simulation(init_pop, fitness, selection_func = naive_selection, crossover= blx_alpha_crossover, mutation = shrink_mutation, gen_nb=100,
               child_nb = 10000, select_nb =10, alpha=0.1, mutation_rate = 0, elites = 2, precision = 4, printer=True, crisis_threshold = 5):
    """
    simulation of genetic evolution on many generations
    :param init_pop: initial pop
    :param fitness: fitness form
    :param func: function evaluation
    :param nb_gen: number of generations
    :param child_nb: number of children per generation
    :param nb_select: number of individuals selected per generation
    :param alpha: blx alpha hyperparameter
    :return: final population
    """
    parents = init_pop
    parents = np.around(parents, precision)
    if printer:
        print("Gen 0 best minimization = {}".format(round(fitness(parents).min(), precision)))
    selected_parents = selection_func(parents, select_nb, fitness)
    for i in range(1, gen_nb+1):
        #all crossed children
        children = crossing(selected_parents, fitness, crossover=crossover, child_nb=child_nb, alpha=alpha)
        #mutation
        children = mutation(children, mutation_rate = mutation_rate)
        #elitism
        children = elitism(selected_parents, children, fitness,  number = elites)
        children = np.around(children, precision)
        if i%10==0 and printer:
            print("Gen {} best minimization = {}".format(i, round(fitness(children).min(), precision)))

        parents = children
        # biological crisis
        if i % (gen_nb // 4) == 1 and fitness(parents).min() > crisis_threshold:
            max_abs = 2*max(abs(init_pop.min()), abs(init_pop.max()))
            # selected_parents += np.random.uniform(-2*max_abs,2*max_abs, selected_parents.shape)
            selected_parents = random_selection(parents, select_nb//4)
            selected_parents += np.random.uniform(-max_abs, max_abs,select_nb//4)
        else:
            # selection
            selected_parents = selection_func(parents, select_nb, fitness)
            selected_parents = elitism(parents, selected_parents, fitness, number = 1)
    return children, children[:, np.argmin(fitness(children))]

def multiple_runs(init_pop, fitness, selection_func = wheel_selection, crossover= uniform_crossover, mutation = shrink_mutation, gen_nb=1500,
               child_nb = 1000, select_nb =10, alpha=0.5, mutation_rate = 0.25, elites = 1, precision = 4, nb_runs = 10, printer=False, crisis_threshold =5):
    minimal_val = np.zeros(nb_runs)
    results = np.zeros((init_pop.shape[0], nb_runs))
    for i in range(nb_runs):
        children, best_child = simulation(init_pop, fitness, selection_func = selection_func, crossover= crossover, mutation = mutation, gen_nb=gen_nb,
               child_nb = child_nb, select_nb =select_nb, alpha=alpha, mutation_rate = mutation_rate, elites = elites, precision = precision, printer=printer, crisis_threshold = crisis_threshold)
        fit_eval = fitness(best_child)
        minimal_val[i] = fit_eval
        results[:,i] = best_child
        print(f"Run {i} best fitness = {fit_eval}")
    avg_val = round(np.mean(minimal_val), precision)
    std_val = round(np.std(minimal_val), precision)
    best_overall = results[:, np.argmin(fitness(results))]
    print("======================================================")
    print(f"{nb_runs} simulations of {gen_nb} generations report:")
    print("Average fitness = ", avg_val)
    print("Average fitness Std = ", std_val)
    print("Best individual overall = ", best_overall)
    return best_overall, results, avg_val, std_val

def pso(X, fitness, func, c0=1,  c1 = 1, c2 = 1, range_val = [-10, 10], nb_iter = 100):
    nb_points = X.shape[0]
    pop_size = X.shape[1]
    V = np.random.uniform(range_val[0], range_val[1], X.shape)/4
    personal_best = np.random.uniform(range_val[0], range_val[1], X.shape)
    fit_eval = fitness(func, X)
    global_best = X[:, np.argmax(fit_eval)][:,None]
    best_fitness = fitness(func, global_best)[0]
    print("Gen 0 best fitness = {}".format(best_fitness))
    for i in range(1, nb_iter+1):
        old_fit_eval = fitness(func, X)
        w = np.random.uniform(range_val[0],range_val[1], pop_size)
        r1 = np.random.uniform(range_val[0],range_val[1], pop_size)
        r2 = np.random.uniform(range_val[0],range_val[1], pop_size)
        cognitive_comp = personal_best - X
        social_comp = global_best - X
        V = c0*w*V + c1*r1*cognitive_comp + c2*r2*social_comp
        new_X = X + V
        new_X[new_X >range_val[1]] = range_val[1]
        new_X[new_X < range_val[0]] = range_val[0]
        new_fit_eval = fitness(func, new_X)
        max_fitness = np.max(new_fit_eval)
        if max_fitness>best_fitness:
            best_fitness = max_fitness
            global_best = X[:, np.argmax(new_fit_eval)][:,None]
        diff = ((new_fit_eval - old_fit_eval)>0).astype('int')
        personal_best = diff*new_X + (1-diff)*X
        X = new_X
        if i%10==0:
            print("gen {} best fitness = {}".format(i,best_fitness))
    plt.figure()
    plt.scatter(new_X[0,:], new_X[1,:])
    plt.show()
    return global_best.flatten()



if __name__ == "__main__":
    # individual size
    n = 10
    # Population size
    N = 100000
    val_range = [-5.12, 5.12]
    val_range = [-100, 100]
    val_range=[-10,10]
    fitness = rastrigin
    fitness = rosenbrock
    #fitness = sphere
    #fitness = booth
    initial_pop = np.random.uniform(val_range[0], val_range[1], (n,N))
    best, results, avg, std = multiple_runs(initial_pop,
                                fitness,
                                selection_func=wheel_selection,
                                mutation=shrink_mutation,
                                crossover=uniform_crossover,
                                gen_nb=800,
                                child_nb=1000,
                                select_nb=100,
                                alpha=0.5,
                                mutation_rate=0.25,
                                elites=1,
                                precision=4,
                                nb_runs = 10,
                                crisis_threshold = 5)
