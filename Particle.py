import numpy as np


class Particle:
    """
    Class for a Particle of a swarm for a Particle Swarm Optimization
    """
    def __init__(self, D, boundary, value):
        """
        Constructor of the class Particle
        :param D: Number of dimension of the search space
        :param boundary: boundaries (lower and upper value) for all the dimension (i.e. list with two values)
        """
        # Position of the particle
        self.__position = np.random.uniform(boundary[0], boundary[1], D)
        # Velocity of the particle
        self.__velocity = np.random.random(D)
        # Best search point the particle has seen (initially set to the initial position)
        self.__bestSP = self.__position
        # Informants initially set to None
        self.__informants = None
        # Value
        self.__fitness_value = value

    def set_informant(self, informants):
        """
        Set the informants of the Particle
        :param informants: the informants to set to the particle (list of instances of Particle class)
        :return:
        """
        self.__informants = informants

    def __str__(self):
        return 'Pos : {0} - Velocity : {1} \nBest Search Point : {2} \nInformants : {3}'.format(self.__position, self.__velocity, self.__bestSP, self.__informants)

    def getPosition(self):
        return self.__position

    def setPosition(self, value):
        self.__position = value

    def getVelocity(self):
        return self.__velocity

    def setVelocity(self, value):
        self.__velocity = value

    def setFitnessValue(self, value):
        if value < self.__fitness_value:
            self.__bestSP = self.__position
        self.__fitness_value = value

    def getFitnessValue(self):
        return self.__fitness_value

    def getInformants(self):
        return self.__informants

    def getBestSP(self):
        return self.__bestSP

if __name__=='__main__':
    part = Particle(3, [-3, 3], 0)
    print(part)