import numpy as np
import argparse
import BenchmarkFunctions
import PSO
from GA_class import *

###############################################################################################
# HYPER-PARAMETERS
###############################################################################################
dimension = 10                              # Dimension of the fitness function
n_pop = 40                                  # Population size
n_iter = 3000                               # Number of maximum iteration

# Particle Swarm Optimization (PSO)
n_inf = 'random'                            # Number of informants for each particle (could be a number or 'random')
opt = ['time_varying_inertia',              # Optimizer : list of adaptive techniques you want to use (among 'time_varying_inertia' and time_varying_acceleration')
       'time_varying_acceleration']         #               if you don't want to use one you need to provide an empty list []
alpha_pso = [0.4, 0.9]                      # Weight for the inertia component
beta = [2.5, 0.5]                           # Weight for the cognitive component
gamma = [0.5, 2.5]                          # Weight for the social component
delta = 0.4                                 # Weight for the best position seen by all the population
epsilon = 1                                 # Step for the update of the position given the velocity

#   Note that if you want to use an adaptive weight for alpha and/or beta and gamma, you need to provide a list of two
#   values for these fields (first the initial value and then the final value).
#     e.g. opt = ['time_varying_inertia']
#          alpha = [0.4, 0.9]



# Genetic Algorithm (GA)
selection = "tournament"                                        # Selection type. Possible choices : "tournament", "roulette", "naive".
                                                                # Naive consists in blindly taking the best individuals.

tournament_size = 10                                            # For Tournament selection only, the tournament size. Should be < n_pop
winner_proba = 0.9                                              # For Tournament selection only, proba p that winner is selected.
                                                                # Otherwise, ranked n of tournament is selected with proba p(1-p)^(n-1)

percentage_pop_selected = 0.2                                   # percentage of pop selected (to be parents of next generation)
how_many_selected = int(min(percentage_pop_selected, 1)*n_pop)  # number of pop selected. No need to edit.

crossover = 'k-points'                                          # Crossover type. Possible choices : 'k-points', 'uniform', 'blx-alpha'
cp = 0.8                                                        # Crossover rate. When crossover doesn't occur, parents simply go to next generation
k_points = 1                                                    # Only for k-points crossover : number of cuts in genome. Should be < dimension//2
alpha = 0.1                                                     # Only for blx-alpha. blx-alpha range of search. Often taken between 0.1 and 0.15

mutation = "shrink"                                             # Mutation type. Possible choices : 'shrink', 'cmaes'
                                                                # 'shrink' corresponds to Gaussian shrinkage
                                                                # 'cmaes' (Covariance Adaptation Search) is an evolutionary search strategy
                                                                # which has been derived here into a mutation.
                                                                # Doesn't totally correspond to a genetic strategy though since it works without crossover.
                                                                # Not investigated in the report (off-topic) but works with all selection types (best with naive) for low number of iterations

mp = 0.05                                                       # Mutation rate
V_0 = 0                                                         # Initial Gaussian variance. 0 triggers the auto-scaled variance mode (Var = max|pop|)
                                                                # Recommended to test with strides of 20

elites = 5                                                      # number of elites. < n_pop
culling = 5                                                     # number of eliminated individuals (least fitted ones). < n_pop

thresh_reset = 150                                              # If min fitness is above thresh_reset at every quarter of n_iter, resets the population as initially






##############################################################################################################
# Execution of the script
def main():
    """
    Main function to use the implementation of the optimization algorithms
    :return:
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--pso", help="Use PSO algorithm with the hyperparameters specified in the main.py file", action="store_true")
    parser.add_argument("--ga", help="Use the Genetic Algorithm (GA) with the hyperparameters specified in the main.py file", action="store_true")
    parser.add_argument("-f1", "--rosenbrock",
                        help="Use Rosenbrock function as fitness function to be minimized with the optimisation algorithm ", action="store_true")
    parser.add_argument("-f2", "--rastrigin",
                        help="Use Rastrigin function as fitness function to be minimized with the optimisation algorithm ", action="store_true")
    parser.add_argument("-f3", "--griewank",
                        help="Use Griewank as fitness function to be minimized with the optimisation algorithm ", action="store_true")
    parser.add_argument("-k", "--k_test", type=int, default=0, help="Use repeted test with the number of test specified by this option")
    parser.add_argument("-p", "--plot", help="Plot the evolution of the best fitness value seen during the optimization algorithm", action="store_true")

    args = parser.parse_args()
    if (not args.pso) and (not args.ga):
        print('Warning : Unspecified optimization algorithm ')
        print('By default, the PSO algorithm will be use')
        args.pso = True
    if (not args.rastrigin) and (not args.rosenbrock) and (not args.griewank):
        print('Warning : unspecified fitness function (possible fitness is Rastrigin, Griewank or Rosenbrock)')
        print('By default, the fitness function is set to Rastrigin')
        args.rastrigin = True
    # Instantiate the fitness function
    boundary, fitness, fitness_string = set_fitness(args)
    if args.pso:
        if args.k_test:
            PSO.repeted_test(fitness, dimension, boundary, args.k_test, n_pop, opt, alpha_pso, beta, gamma, delta, epsilon, n_iter, n_inf)
        else:
            pso = PSO.PSO(n_particles=n_pop, D=dimension, boundary=boundary, init_value=1000, opt=opt)
            pso.set_hyperparameters(alpha_pso, beta, gamma, delta, epsilon)
            pso.set_fitness(fitness)
            pso.optimization_algorithm(n_iter=n_iter, n_informants=n_inf)
            if args.plot:
                pso.result_curve()
    # GA
    else :
        experience = GA(nb_genes=dimension, initial_pop_size=n_pop, bound_min=boundary[0], bound_max=boundary[1], fitness_name=fitness_string, elites=elites, cull=culling)
        experience.selection_init(number=how_many_selected, name=selection, tournament_size=tournament_size, proba=winner_proba)
        if crossover == "k-points":
            experience.crossover_init(child_nb=n_pop, name=crossover, alpha=min(k_points, dimension//2), rate=cp)
        elif crossover == "blx-alpha":
            experience.crossover_init(child_nb=n_pop, name=crossover, alpha=alpha, rate=cp)
        # Uniform
        else :
            experience.crossover_init(child_nb=n_pop, name="uniform", alpha=0.5, rate=cp)
        experience.mutation_init(name=mutation, width=V_0, rate=mp)
        if args.k_test:
            best, results, avg, std = experience.multiple_runs(nb_runs=args.k_test, nb_gen=n_iter, precision=6, printer=1, thresh_reset = thresh_reset)
        else:
            best, results, avg, std = experience.multiple_runs(nb_runs=1, nb_gen=n_iter, precision=6, printer=1, thresh_reset = thresh_reset)
            if args.plot:
                print("Sorry, no plot available for GA yet.")


def set_fitness(args):
    """
    Instantiate the fitness function chose by the user. If multiple fitness functions have been entered, only Rastrigin will
    be use by default
    """
    if args.rastrigin:
        boundary = [-5.12, 5.12]
        fitness = BenchmarkFunctions.Rastrigin(n_dim = dimension, boundary=boundary)
        return boundary, fitness, "rastrigin"
    elif args.rosenbrock:
        boundary = [-100, 100]
        fitness = BenchmarkFunctions.Rosenbrock(n_dim=dimension, boundary=boundary)
        return boundary, fitness, "rosenbrock"
    else:
        boundary = [-600, 600]
        fitness = BenchmarkFunctions.Griewank(n_dim=dimension, boundary=boundary)
        return boundary, fitness, "griewank"


if __name__=='__main__':
    main()



