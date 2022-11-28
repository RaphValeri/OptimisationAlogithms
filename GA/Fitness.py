import numpy as np

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
    Sphere function (shifted by [0, 1, 2, ...])
    :param x: 1D array of size >2 (numpy : shape = (n,1))
    :return: sphere evaluation
    """
    shift = np.array([[i] for i in range(x.shape[0])])
    X = x - shift
    S = np.sum(X**2, axis = 0)
    return S

def double_sum(x):
    s1 = np.array([np.sum(x[:(i+1),:], axis = 0) for i in range(len(x))])
    s2 = np.sum(s1**2, axis = 0)
    return s2

def schwefel(x):
    V = 418.9829
    S = V*x.shape[0] - np.sum(x*np.sin(np.sqrt(abs(x))), axis = 0)
    return S

def griewank(x):
    S = 1 + np.sum((x**2), axis =0)/4000 - np.prod(np.cos(np.array([[1/np.sqrt(i)] for i in range(1, x.shape[0]+1)])*x), axis = 0)
    return S

def step(x):
    #S = np.sum(abs(x.astype('int')), axis =0)
    S = np.min(abs(x), axis=0)
    return S

class Fitness:
    def __init__(self, name="sphere"):
        self.possible_names = {"sphere":sphere,
                               "ds": double_sum,
                               "rastrigin":rastrigin,
                               "rosenbrock":rosenbrock,
                               "schwefel": schwefel,
                               "griewank":griewank,
                               "step":step
                               }
        if name not in self.possible_names.keys() :
            print("Warning, fitness name is not recognized")
            print("Admissible fitness names are : 'sphere', 'rastrigin', 'rosenbrock'")
            print("sphere fitness is taken by default")
            self.name = "sphere"
        else :
            self.name = name
        self.evaluate = self.possible_names[self.name]

    def __str__(self):
        string =  "{} fitness".format(self.name)