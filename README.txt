This repository concerns the implementation of two well-known optimization algorithms PSO (Particle Swarm
Optimization) and GA (Genetic Algorithm) from scratch (without using specific optimization library)
designed during a coursework of the Biologically Inspired Computation course at Heriot-Watt University.

Authors : Alexandre Kha and Raphaël Valeri
Date : 28th November 2022

To run the code you can execute the following command
    python main.py
    This will run the PSO algorithm on the Rastrigin function with default values for the hyperparameters

To access the documentation of this execution you can run the following command:
    python main.py -h

If you want to use the GA you can run with the option --ga
    python main.py --ga
    python main.py --pso for PSO

If you want to use one specific fitness function among Rosenbrock, Rastrigin and Griewank you need to add
respectively the option f1, f2 or f3
e.g.
    python main.py --ga -f1

Other options are available:
    -k with a number to run k-repeated run
    -p to plot the evolution of the optimization by the PSO algorithm if used

If you want to modify the values of the hyperparameters, you need to change their values at the beginning of the
main.py script (first, the values for the PSO are set and then the ones of GA)

Note that you can visualize the swarm of a PSO algorithm in a continuous contour plot when the dimension is 2. YOu can try to
replace the value of the field 'dimension' in the main.py file by two.
