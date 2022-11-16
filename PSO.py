import numpy as np
import matplotlib.pyplot as plt
import datetime
import Particle
from BenchmarkFunctions import BenchmarkFunction, Rastrigin, Rosenbrock, Sphere, Schwefel, Griewank


class PSO:

    def __init__(self, n_particles, D, boundary, init_value, opt=None):
        # PSO Problem
        self.__fitness = None
        self.__dim = D
        self.__boundary = boundary
        # Hyper parameters
        self.__opt = opt
        self.__alpha_min = None
        self.__alpha_max = None
        self.__alpha = None
        self.__beta = None
        self.__beta_i = None
        self.__beta_f = None
        self.__gamma = None
        self.__gamma_i = None
        self.__gamma_f = None
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
        if len(self.__opt)==2:
            if self.__opt[0] == 'time_varying_inertia':
                self.__alpha_min = alpha[0]
                self.__alpha_max = alpha[1]
            if self.__opt[1] == 'time_varying_acceleration':
                self.__beta_i = beta[0]
                self.__beta_f = beta[1]
                self.__gamma_i = gamma[0]
                self.__gamma_f = gamma[1]
        elif len(self.__opt) == 1:
            if self.__opt[0] == 'time_varying_inertia':
                self.__alpha_min = alpha[0]
                self.__alpha_max = alpha[1]
            else:
                self.__alpha = alpha
            if self.__opt[0] == 'time_varying_acceleration':
                self.__beta_i = beta[0]
                self.__beta_f = beta[1]
                self.__gamma_i = gamma[0]
                self.__gamma_f = gamma[1]
            else:
                self.__beta = beta
                self.__gamma = gamma
        else:
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

    def update_velocity(self, best_pos, n_iter_max, n_iter):
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
            # Check to use or not optimizers
            if self.__opt is not None:
                if 'time_varying_inertia' in self.__opt:
                    # Update the value of the inertia weight
                    self.time_varying_inertia_weight(n_iter_max, n_iter)

                if 'time_varying_acceleration' in self.__opt:
                    # Update the value of the acceleration coefficients
                    self.time_varying_acc_coefs(n_iter_max, n_iter)

            for n in range(self.__dim):
                b = self.__beta*np.random.random()
                c = self.__gamma*np.random.random()
                d = self.__delta * np.random.random()
                inertia = self.__alpha*part.getVelocity()[n]
                cognitive = b*part.getBestSP()[n]-b*part.getPosition()[n]
                social = c*best_inf.getPosition()[n]-c*part.getPosition()[n]
                new_velocity[n] = inertia + cognitive + social + d*(best_pos[n]-part.getPosition()[n])
            part.setVelocity(new_velocity)

    def time_varying_inertia_weight(self, n_iter_max, n_iter):
        """
        Time varying inertia weight from
        :param n_iter_max: the maximum of the iteration number
        :param n_iter: current iteration number
        :return:
        """
        self.__alpha = (n_iter_max-n_iter)/n_iter_max*(self.__alpha_max-self.__alpha_min)+self.__alpha_min


    def time_varying_acc_coefs(self, n_iter_max, n_iter):
        """
        Time varying acceleration coefficients from A. Ratnaweera, S. K. Halgamuge and H. C. Watson,
        "Self-organizing hierarchical particle swarm optimizer with time-varying acceleration coefficients,"
        in IEEE Transactions on Evolutionary Computation.
        :param n_iter_max: the maximum of the iteration number
        :param n_iter: current iteration number
        :return:
        """
        self.__beta = (self.__beta_i - self.__beta_f)*(n_iter_max - n_iter)/n_iter_max + self.__beta_f
        self.__gamma = (self.__gamma_i - self.__gamma_f) * (n_iter_max - n_iter) / n_iter_max + self.__gamma_f


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
            self.update_velocity(best_position, n_iter_max = n_iter, n_iter=n)
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
            if ((n_iter >= 2000) and ((n % 500) == 0 or n == n_iter-1)) or ((n_iter < 2000) and ((n % 5) == 0 or n == n_iter-1)):
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
        if self.__opt is not None :
            if 'time_varying_inertia' in self.__opt:
                alpha = r'$\alpha \in$ [{},{}]'.format(self.__alpha_min, self.__alpha_max)
            else:
                alpha = r'$\alpha={}$'.format(self.__alpha)
            if 'time_varying_acceleration' in self.__opt:
                beta = r'$\beta \in$ [{},{}]'.format(self.__beta_i, self.__beta_f)
                gamma = r'$\gamma \in$ [{},{}]'.format(self.__gamma_i, self.__gamma_f)
            else:
                beta = r'$\beta={}$'.format(self.__beta)
                gamma = r'$\gamma={}$'.format(self.__gamma)
        else:
            alpha = r'$\alpha={}$'.format(self.__alpha)
            beta = r'$\beta={}$'.format(self.__beta)
            gamma = r'$\gamma={}$'.format(self.__gamma)
        title = r'PSO Algorithm used to optimize {} function (n_population={}, {}, {}, {}, $\delta={}$, $\epsilon={}$)'.format(self.__fitness.getName(), len(self.__population), alpha, beta, gamma, self.__delta, self.__eps)
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
    boundary = [0, 600]
    fitness = Griewank(n_dim, boundary, noise=True)

    # HYPER-PARAMETERS
    eps = 1.0               # step for the update of the position given the velocity
    alpha = 1.0             # weight for the previous velocity (inertia component)
    beta = 1.8              # weight for the cognitive component
    gamma = 1.8             # weight for the social component
    delta = 0.4             # weight for the best position seen by all the population

    # TIME VARYING HYPER-PARAMETERS
    alpha_opt = [0.4, 0.9]  # For time varying inertia weight (TVIW)
    beta_opt = [2.5, 0.5]   # For time varying acceleration coefficient (TVAC)
    gamma_opt = [0.5, 2.5]  # For time varying acceleration coefficient(TVAC)

    # PSO Algorithm
    pso = PSO(n_particles=75, D=n_dim, boundary=boundary, init_value=100, opt=['time_varying_inertia', 'time_varying_acceleration'])
    pso.set_hyperparameters(alpha_opt, beta_opt, gamma_opt, delta, eps)
    pso.set_fitness(fitness)
    pso.optimization_algorithm(100, 'random')