import numpy as np
import argparse
import BenchmarkFunctions
import PSO

###############################################################################################
# HYPER-PARAMETERS
###############################################################################################
dimension = 10                              # Dimension of the fitness function
n_pop = 40                                  # Population size
n_iter = 3000                               # Number of maximum iteration

# Particle Swarm Optimization (PSO)
n_inf = 'random'                            # Number of informants for each particle (could be a number or 'random')
opt = ['time_varying_inertia',              # Optimizer : list of adaptive techniques you want to use
       'time_varying_acceleration']
alpha = [0.4, 0.9]                          # Weight for the inertia component
beta = [2.5, 0.5]                           # Weight for the cognitive component
gamma = [0.5, 2.5]                          # Weight for the social component
delta = 0.4                                 # Weight for the best position seen by all the population
epsilon = 1                                 # Step for the update of the position given the velocity

# Genetic Algorithm (GA)
#TODO : Add hyperparamters of GA


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
    boundary, fitness = set_fitness(args)
    if args.pso:
        pso = PSO.PSO(n_particles=n_pop, D=dimension, boundary=boundary, init_value=1000, opt=opt)
        pso.set_hyperparameters(alpha, beta, gamma, delta, epsilon)
        pso.set_fitness(fitness)
        pso.optimization_algorithm(n_iter=n_iter, n_informants=n_inf)
        if args.plot:
            pso.result_curve()


def set_fitness(args):
    """
    Instantiate the fitness function chose by the user. If multiple fitness functions have been entered, only Rastrigin will
    be use by default
    """
    if args.rastrigin:
        boundary = [-5.12, 5.12]
        fitness = BenchmarkFunctions.Rastrigin(n_dim = dimension, boundary=boundary)
        return boundary, fitness
    elif args.rosenbrock:
        boundary = [-100, 100]
        fitness = BenchmarkFunctions.Rosenbrock(n_dim=dimension, boundary=boundary)
        return boundary, fitness
    else:
        boundary = [-600, 600]
        fitness = BenchmarkFunctions.Griewank(n_dim=dimension, boundary=boundary)
        return boundary, fitness


if __name__=='__main__':
    main()



