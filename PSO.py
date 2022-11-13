import numpy as np
import matplotlib.pyplot as plt
import datetime
import Particle
from BenchmarkFunctions import BenchmarkFunction, Rastrigin, Rosenbrock, Sphere, Schwefel


class PSO:

    def __init__(self, n_particles, D, boundary, init_value):
        # PSO Problem
        self.__fitness = None
        self.__dim = D
        self.__boundary = boundary
        # Hyper parameters
        self.__alpha = None
        self.__beta = None
        self.__gamma = None
        self.__delta = None
        self.__eps = None
        # Population (Swarm) of the PSO
        self.__population = np.array([Particle.Particle(D, boundary, init_value) for i in range(n_particles)])
        # History of the best seen solution
        self.__history = {'fitness_value': [], 'position': [], 'iteration': []}

    def set_hyperparameters(self, alpha, beta, gamma, delta, eps):
        """
        Set the value of the hyperparameters used in the PSO algorithm
        :param alpha: weight for the previous velocity
        :param beta: weight for the cognitive component
        :param gamma: weight for the social component
        :param delta: weight for the best position seen by all the population
        :param eps: step for the update of the position given the velocity
        :return:
        """
        self.__alpha = alpha
        self.__beta = beta
        self.__gamma = gamma
        self.__delta = delta
        self.__eps = eps

    def set_fitness(self, fitness_function):
        """
        Set the fitness function of the problem
        :param fitness_function: function to optimize with PSO (instance of BenchmarkFunction class)
        :return:
        """
        self.__fitness = fitness_function

    def update_informants(self, n_informants='random'):
        """
        Update the informants for each Particle of the population
        :param n_informants: number of informants to set for each Particle (can be set to 'random' ron use random number)
        :return:
        """
        if n_informants == 'random':
            n_inf = np.random.randint(1, len(self.__population))
        else:
            n_inf = n_informants
        for i in range(len(self.__population)):
            # Indices of Particle which can be set to an informants for the Particle i (all except Particle i)
            idx_possible = [x for x in range(0, len(self.__population))]
            idx_possible.remove(i)
            # Random choices of the informants given the number of informants n_inf
            idx_informants = np.random.choice(idx_possible, size=n_inf)
            self.__population[i].set_informant(self.__population[idx_informants])

    def update_velocity(self, best_pos):
        """
        Update the velocity of each Particle
        :param best_pos: best position seen by any member of the population
        :return:
        """
        for i in range(len(self.__population)):
            part = self.__population[i]
            # Sort the list of the informants ( in which we add the particle) in relation
            # to the fitness value of the best search point of the informants
            informants = list(part.getInformants())
            informants.append(part)
            informants = sorted(informants, key=lambda x: self.__fitness.value(x.getBestSP()))
            best_inf = informants[0]
            # Update the velocity for each dimension using random weights
            new_velocity = np.zeros(self.__dim)
            for n in range(self.__dim):
                b = self.__beta*np.random.random()
                c = self.__gamma*np.random.random()
                d = self.__delta * np.random.random()
                inertia = self.__alpha*part.getVelocity()[n]
                cognitive = b*part.getBestSP()[n]-b*part.getPosition()[n]
                social = c*best_inf.getPosition()[n]-c*part.getPosition()[n]
                new_velocity[n] = inertia + cognitive + social + d*(best_pos[n]-part.getPosition()[n])
            part.setVelocity(new_velocity)

    def optimization_algorithm(self, n_iter, n_informants):
        """
        Algorithm of the PSO optimization
        :param n_iter: number of iterations
        :param n_informants: number of informants to set for each Particle (can be set to 'random' ron use random number)
        :return:
        """
        t0 = datetime.datetime.now()
        best_position = self.__population[np.random.randint(0, len(self.__population))].getPosition()
        for n in range(n_iter):
            self.update_informants(n_informants)
            # Fitness value of each Particle
            for i in range(len(self.__population)):
                # Evaluation of the Particle i
                part = self.__population[i]
                # Update of fitness value and the best SP of the particle
                part.setFitnessValue(self.__fitness.value(part.getPosition()))
                # Update of the best Particle
                if part.getFitnessValue() < self.__fitness.value(best_position):
                    best_position = part.getPosition()
            # Update velocity
            self.update_velocity(best_position)
            # Update Particles' position
            for i in range(len(self.__population)):
                part = self.__population[i]
                new_pos = part.getPosition() + self.__eps*part.getVelocity()
                # Check the new position is in the search space
                for j in range(self.__dim):
                    if new_pos[j] < self.__boundary[0]:
                        new_pos[j] = self.__boundary[0]
                    elif new_pos[j] > self.__boundary[1]:
                        new_pos[j] = self.__boundary[1]
                part.setPosition(new_pos)
            if ((n_iter >= 2000) and ((n % 500) == 0 or n == n_iter-1)) or ((n_iter < 2000) and ((n % 30) == 0 or n == n_iter-1)):
                print('---'*10)
                print('Iteration {} - Best value = {} for position = {}'.format(n, self.__fitness.value(best_position), best_position))
                # Update the history
                self.__history['iteration'].append(n)
                self.__history['position'].append(best_position)
                self.__history['fitness_value'].append(self.__fitness.value(best_position))
            # Continuous 2D plot for 2D problem (contour visualization)
            #if self.__dim == 2:
                #self.__fitness.plot_2D(best_position, [self.__population[i].getPosition() for i in range(len(self.__population))])

        # 3D plot at the end of the algorithm for 2D problem
        if self.__dim == 2:
            self.__fitness.plot_3D(best_position, [self.__population[i].getPosition() for i in range(len(self.__population))])
        tf = datetime.datetime.now() - t0
        print('Optimization time : {}'.format(tf))
        self.result_curve()
        return self.__history

    def result_curve(self):
        """
        Plot the evolution of the best found solution (fitness value and the position for each dimension)
        :return:
        """
        fig, ax = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
        title = r'PSO Algorithm used to optimize {} function (n_population={}, $\alpha={}$, $\beta={}$, $\gamma={}$, $\delta={}$, $\epsilon={}$)'.format(self.__fitness.getName(), len(self.__population), self.__alpha, self.__beta, self.__gamma, self.__delta, self.__eps)
        fig.suptitle(title, fontsize=16, fontweight='bold')
        # Plot the evolution of the best fitness value
        ax[0].set_title('Evolution of the best fitness value')
        ax[0].plot(self.__history['iteration'], self.__history['fitness_value'], linestyle='dashed', marker='o')
        ax[0].grid()
        ax[0].set_xlabel('Iteration')
        ax[0].set_ylabel('Best fitness value (lower is the best)', rotation=90)
        # Plot the evolution of the position of the best Search Point
        ax[1].grid()
        position_history = np.array(self.__history['position'])
        for n in range(self.__dim):
            ax[1].plot(self.__history['iteration'], position_history[:, n], label='dimension {}'.format(n+1), linestyle='dashed', marker='o')
        ax[1].set_xlabel('Iteration')
        ax[1].set_ylabel('Position value')
        ax[1].set_title('Evolution of the best seen position')
        ax[1].legend()
        plt.show()


if __name__ == '__main__':
    # PROBLEM
    n_dim = 10
    boundary = [-5.12, 5.12]
    fitness = Rastrigin(n_dim, boundary)

    # HYPER-PARAMETERS
    eps = 1.0    # step for the update of the position given the velocity
    alpha = 1  # weight for the previous velocity (inertia component)
    beta = 3   # weight for the cognitive component
    gamma = 0.1  # weight for the social component
    delta = 0.8  # weight for the best position seen by all the population

    # PSO Algorithm
    pso = PSO(n_particles=150, D=n_dim, boundary=boundary, init_value=100)
    pso.set_hyperparameters(alpha, beta, gamma, delta, eps)
    pso.set_fitness(fitness)
    pso.optimization_algorithm(15000, 'random')