import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from abc import ABC, abstractmethod
import copy


class BenchmarkFunction(ABC):
    """
    Abstract cass for the Benchmark functions implemented to be tested with the PSO and GA algorithms
    """

    def __init__(self, n_dim, boundary, name):
        """
        Constructor
        :param n_dim: number of dimension of the function
        :param boundary: the boundary (need to be a list with the lower and upper value e.g. [-5, 5])
        :param name: name of the function
        """
        self.__n_dim = n_dim
        self.__boundary = boundary
        self.__name = name

    def getName(self):
        return self.__name

    @abstractmethod
    def value(self, x):
        pass

    @abstractmethod
    def value_2D(self, x, y):
        pass

    def plot_2D(self, x, population):
        """
        PLot the countour of the function (if its dimension is 2) with dynamic plot of the population and the better seen position
        :param x: better seen position
        :param population: list of the position of each member of the population
        :return:
        """
        if self.__n_dim !=2:
            print('Warning ! You cannot visualize the fitness function in a dimension higher than 2')
            return

        plt.clf()
        # Make data.
        X = np.arange(self.__boundary[0], self.__boundary[1], 0.05)
        Y = np.arange(self.__boundary[0], self.__boundary[1], 0.05)
        X, Y = np.meshgrid(X, Y)
        Z = self.value_2D(X, Y)

        plt.contour(X, Y, Z)

        for i in range(len(population)):
            plt.scatter(population[i][0], population[i][1], color='k')
        plt.scatter(x[0], x[1], color='r', label='Best seen position')
        plt.xlim(self.__boundary)
        plt.ylim(self.__boundary)
        plt.title('{} function optimization'.format(self.__name))
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.legend()
        #plt.show()
        plt.pause(0.5)

    def plot_3D(self, x, population):
        """
        Plot the function in 3 dimensions (if its dimension is 2) with the polt of the populationa nd the better seen position
        :param x: Better seen position
        :param population: list of the position of each member of the population
        :return:
        """
        if self.__n_dim !=2:
            print('Warning ! You cannot visualize the fitness function in a dimension higher than 2')
            return

        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

        # Make data.
        X = np.arange(self.__boundary[0], self.__boundary[1], 0.05)
        Y = np.arange(self.__boundary[0], self.__boundary[1], 0.05)
        X, Y = np.meshgrid(X, Y)
        Z = self.value_2D(X, Y)


        # Plot the surface.
        surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, antialiased=False)

        #Plot the population
        for i in range(len(population)):
            ax.scatter3D(population[i][0], population[i][1],self.value(population[i]), color='k')
        #Plot the best position
        ax.scatter3D(x[0], x[1], self.value(x), color='r', label='Best seen position')

        #ax.contour(X, Y, Z, 10, offset=-1, colors="k", linestyles="solid", alpha=0.5)

        plt.xlim(self.__boundary)
        plt.ylim(self.__boundary)
        plt.title('{} function optimization'.format(self.__name))
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.legend()
        plt.show()
        plt.pause(0.4)


class Rastrigin(BenchmarkFunction):
    """
    Rastrigin benchmark function
    """
    def __init__(self, n_dim, boundary):
        super().__init__(n_dim, boundary, 'Rastrigin')
        self.__n_dim = n_dim
        self.__boundary = boundary

    def value(self, x):
        A = 10
        X = copy.deepcopy(x)
        if isinstance(X, list):
            X = np.array(X)
        if X.shape[0] != self.__n_dim:
            X = X.T
        if len(X.shape) < 2:
            X = X.reshape((self.__n_dim, 1))
        S = np.sum(X ** 2 - A * np.cos(2 * np.pi * X), axis=0)
        return A * self.__n_dim + S

    def value_2D(self, x, y):
        return 10*self.__n_dim + x**2 + y**2 - 10*(np.cos(2*np.pi*x)+np.cos(2*np.pi*y))

class Rosenbrock(BenchmarkFunction):
    """
    Rosenbrock benchmark function
    """
    def __init__(self, n_dim, boundary):
        super().__init__(n_dim, boundary, 'Rosenbrock')
        self.__n_dim = n_dim
        self.__boundary = boundary

    def value(self, x):
        X = copy.deepcopy(x)
        if isinstance(X, list):
            X = np.array(X)
        if X.shape[0] != self.__n_dim:
            X = X.T
        if len(X.shape) < 2:
            X = X.reshape((self.__n_dim, 1))
        xi = X[:-1, :]
        xp1 = X[1:, :]
        S = np.sum(100 * (xp1 - xi ** 2) ** 2 + (1 - xi) ** 2, axis=0)
        return S

    def value_2D(self, x, y):
        return 100*(x-y**2)**2 + (1-y)**2


class Sphere(BenchmarkFunction):
    """
    Sphere benchmark function
    """
    def __init__(self, n_dim, boundary):
        super().__init__(n_dim, boundary, 'Rosenbrock')
        self.__n_dim = n_dim
        self.__boundary = boundary

    # def value(self, x):
    #     return np.sum(x**2)

    def value(self, x):
        X = copy.deepcopy(x)
        if isinstance(X, list):
            X = np.array(X)
        if X.shape[0] != self.__n_dim:
            X = X.T
        if len(X.shape) < 2:
            X = X.reshape((self.__n_dim, 1))
        S = np.sum(X**2, axis=0)
        return S

    def value_2D(self, x, y):
        return x**2+y**2

class Schwefel(BenchmarkFunction):
    """
    Schwefel benchmark function
    """
    def __init__(self, n_dim, boundary, noise=True):
        super().__init__(n_dim, boundary, 'Schwefel')
        self.__n_dim = n_dim
        self.__boundary = boundary
        self.__noise = noise

    def value(self, x):
        X = copy.deepcopy(x)
        if isinstance(X, list):
            X = np.array(X)
        if X.shape[0] != self.__n_dim:
            X = X.T
        if len(X.shape) < 2:
            X = X.reshape((self.__n_dim, 1))
        if self.__noise:
            noise = np.abs(np.random.randn())
        else:
            noise = 0
        s1 = np.array([np.sum(X[:(i + 1), :], axis=0) for i in range(len(X))])
        s2 = np.sum(s1 ** 2, axis=0)
        return s2*(1+0.4*noise)

    def value_2D(self, x, y):
        if self.__noise:
            noise = np.abs(np.random.randn())
        else:
            noise = 0
        cnt = (x**2 + (x+y)**2)
        return cnt*(1+0.4*noise)

class Griewank(BenchmarkFunction):
    """
    Griewank benchmark function
    """
    def __init__(self, n_dim, boundary, noise=False):
        super().__init__(n_dim, boundary, 'Griewank')
        self.__n_dim = n_dim
        self.__boundary = boundary
        self.__noise = noise


    def value(self, x):
        X = copy.deepcopy(x)
        if isinstance(X, list):
            X = np.array(X)
        if X.shape[0] != self.__n_dim:
            X = X.T
        if len(X.shape) < 2:
            X = X.reshape((self.__n_dim, 1))
        S = 1 + np.sum((X ** 2), axis=0) / 4000 - np.prod(np.cos(np.array([[1 / np.sqrt(i)] for i in range(1, x.shape[0] + 1)]) * X), axis=0)
        return S

    def value_2D(self, x, y):
        if self.__noise:
            noise = np.abs(np.random.randn())
        else:
            noise = 0
        x = x*(1+0.3*noise)
        y = y*(1+0.3*noise)
        cnt = (x**2 + y**2)/4000
        prod = np.cos(x/1) + np.cos(y/2)
        return cnt - prod + 1
