import numpy as np
from scipy.linalg import sqrtm

def indicator(min, max, x):
    return (x>=min and x<=max)

def sum_mult(A):
    n = A.shape[0]
    mu = A.shape[1]
    S = np.zeros((n,n))
    for i in range(mu):
        S += A[:,i][:,None]@A[:,i][:, None].T
    return S

def shrink_mutation(X, width, mutation_rate = 0):
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
    if width == 0:
        adder = np.random.normal(loc = 0, scale = np.max(abs(X)), size=X.shape)
    else :
        adder = np.random.normal(loc=0, scale=np.max(abs(X)), size=X.shape)
        #adder = np.random.multivariate_normal(np.zeros(nb_points), X@X.T, nb_indiv).T
    mutated = X + bin_factor*adder
    return mutated

def cmaes_mutation(X, cmaes_param, idx):
    idx = idx.astype('int')
    n = X.shape[0]
    lamb = X.shape[1]
    mu = len(idx)
    if len(cmaes_param)==0:
        cmaes_param["lambda"] = lamb
        cmaes_param["mu"] = mu
        cmaes_param["sigma"] = 1
        cmaes_param["n"] = n
        cmaes_param["C"] = np.identity(n)
        cmaes_param["pc"]= np.zeros((n, 1))
        cmaes_param["psig"] = np.zeros((n, 1))
        cmaes_param["csig"] = 3 / n
        cmaes_param["dsig"] = 0.99
        cmaes_param["cc"] = 4 / n
        cmaes_param["c1"] = 2 / n ** 2
        cmaes_param["cmu"] = mu / n ** 2
        cmaes_param["alpha"] = 1.5
        cmaes_param["E"] = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n ** 2))
        cmaes_param["m"] = np.random.uniform(X.min(), X.max(), (n,1))
        cmaes_param["y"] = np.random.multivariate_normal(np.zeros(n), cmaes_param["C"], cmaes_param["lambda"]).T
        cmaes_param["theta"] = cmaes_param["m"] + cmaes_param["sigma"] * cmaes_param["y"]
    else:
        best_y = cmaes_param["y"][:, idx]
        y_bar = (1 / cmaes_param["mu"]) * np.sum(best_y, axis=1)[:, None]
        mp = cmaes_param["m"] + cmaes_param["sigma"] * y_bar
        cmaes_param["psig"] = (1 - cmaes_param["csig"]) * cmaes_param["psig"] + np.sqrt(
            cmaes_param["csig"]*(2-cmaes_param["csig"])) * np.sqrt(cmaes_param["mu"]) * sqrtm(np.linalg.inv(cmaes_param["C"])) @ \
                              y_bar
        sigma_p = cmaes_param["sigma"] * np.exp((cmaes_param["csig"] / cmaes_param["dsig"]) * (
                    (np.linalg.norm(cmaes_param["psig"]) / cmaes_param["E"]) - 1))
        cmaes_param["pc"] = (1 - cmaes_param["cc"]) * cmaes_param["pc"] + indicator(0, cmaes_param["alpha"] * np.sqrt(
            cmaes_param["n"]), np.linalg.norm(cmaes_param["psig"])) * np.sqrt(
            cmaes_param["cc"]*(2-cmaes_param["cc"])) * np.sqrt(cmaes_param["mu"]) * y_bar
        cmaes_param["C"] = (1 - cmaes_param["c1"] - cmaes_param["cmu"]) * cmaes_param["C"] + \
                           cmaes_param["c1"] * cmaes_param["pc"] @ cmaes_param["pc"].T + cmaes_param["cmu"] * (
                                       1 / cmaes_param["mu"])  * sum_mult(best_y)
        cmaes_param["m"] = mp
        cmaes_param["sigma"] = sigma_p
        cmaes_param["y"] = np.random.multivariate_normal(np.zeros(n), cmaes_param["C"], cmaes_param["lambda"]).T
        cmaes_param["theta"] = cmaes_param["m"] + cmaes_param["sigma"] * cmaes_param["y"]
    return cmaes_param["theta"]

class Mutation:
    def __init__(self, name="shrink", initial_std_width = 1, rate = 0):
        self.possible_mutations = {"shrink": shrink_mutation,
                                   "cmaes" : cmaes_mutation,
                                   }
        if name not in self.possible_mutations.keys():
            print("Warning, mutation name is not recognized")
            print("Admissible mutation names are : 'shrink', 'cmaes'")
            print("shrink is taken by default")
            self.name = "shrink"
        else:
            self.name = name
        self.rate = rate
        self.cmaes_param = {}
        self.initial_std_width = initial_std_width

    def apply(self, population, width = 1, idx = None):
        if self.name == "cmaes":
            return cmaes_mutation(population, self.cmaes_param,  idx)
        else :
            return shrink_mutation(population, width, self.rate)

    def reinitialize_maes(self):
        self.cmaes_param = {}