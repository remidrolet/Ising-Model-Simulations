import numpy as np
import signac
from flow import FlowProject
from flow import directives
import flow.environments
import json
import os


class Project(FlowProject):
    pass

@Project.label
def preint(job):
    return True

@Project.label
def initialized(job):
    return os.path.isfile(job.fn("init.npy"))

# @Project.operation.with_directives({'walltime': 1, 'nranks': 1})
@Project.operation
@Project.post(initialized)
def initialize(job):
    Lambda = np.zeros((job.sp.L,job.sp.L), dtype=np.int16)

    for i in range(Lambda.shape[0]):
        for j in range(Lambda.shape[1]):
            Lambda[i,j] = 1#np.random.choice([-1,1])

    magnetization = [np.sum(Lambda)]
    M_over_time = []

    
    def find_neighbors(x, array):
        neighbors = []
        for i in range(-1,2,2):
            neighbors.append(array[(x[0]+i)%array.shape[0], x[1]])
        for j in range(-1,2,2):
            neighbors.append(array[x[0], (x[1]+j)%array.shape[1]])
        return np.array(neighbors)


    def energy_difference(x, array):
        dE = 2 * array[x] * np.sum(find_neighbors(x, array))
        return dE


    def monte_carlo_step(lattice):
        # choose a random position on lattice
        x = (np.random.randint(lattice.shape[0]), np.random.randint(lattice.shape[1]))

        # calculate energy difference
        dE = energy_difference(x, lattice)

        # If energy difference negative, accept change.  Otherwise, accept with transition probability.  
        if dE <= 0:
            lattice[x] = -1 * lattice[x]
            magnetization[0] = magnetization[0] + 2 * lattice[x]
        else:
            acceptance_prob = np.exp(-dE/job.sp.kT)
            if np.random.rand() < acceptance_prob:
                lattice[x] = -1 * lattice[x]
                magnetization[0] = magnetization[0] + 2 * lattice[x]


    def run_sim(num_steps, lattice):
        for step in range(num_steps):
            if step%job.sp.log_period == 0:
                M_over_time.append(magnetization[0])
            monte_carlo_step(lattice)


    def two_point_disconnected_4pts(r, lattice):
        two_points = []
        for i in range(lattice.shape[0]):
            for j in range(lattice.shape[1]):
                for k in range(-1,2,2):
                    two_points.append(lattice[i,j] * lattice[(i + k*r)%lattice.shape[0], j])
                for l in range(-1,2,2):
                    two_points.append(lattice[i,j] * lattice[i, (j + l*r)%lattice.shape[1]])
        two_points = np.array(two_points)
        
        return np.sum(two_points) / len(two_points)

    run_sim(5*job.sp.init_step, Lambda)

    with open(job.fn('init.npy'), 'wb') as f:
        np.save(f, Lambda)
    with open(job.fn('init_magnetizations.npy'), 'wb') as f:
        np.save(f, M_over_time)


@Project.label
def init_plotted(job):
    return job.isfile(job.fn('mag.png'))

# @Project.operation.with_directives({'walltime': 1, 'nranks': 1})
@Project.operation
@Project.pre.after(initialize)
@Project.post(init_plotted)
def init_plot(job):
    import matplotlib.pyplot as plt

    with open(job.fn('init.npy'), 'rb') as f:
        Lambda = np.load(f)
    with open(job.fn('init_magnetizations.npy'), 'rb') as f:
        M_over_time = np.load(f)

    plt.plot(np.arange(len(M_over_time)), np.array(M_over_time)/job.sp.L**2)
    plt.ylim(-0.1,1.1)
    plt.ylabel("Magnetization")
    plt.xlabel("Markov steps / 10000")
    plt.savefig(job.fn("mag.png"))

    plt.figure(figsize=(20,20), dpi= 500)
    plt.matshow(Lambda)
    plt.savefig(job.fn("matrix.png"))

    plt.close('all')


# @Project.operation.with_directives({'walltime': 1, 'nranks': 1})
@Project.operation
@Project.pre.after(initialize)
def continue_init(job):
    
    with open(job.fn('init.npy'), 'rb') as f:
        Lambda = np.load(f)
    with open(job.fn('init_magnetizations.npy'), 'rb') as f:
        M_over_time = list(np.load(f))


    magnetization = [np.sum(Lambda)]

    
    def find_neighbors(x, array):
        neighbors = []
        for i in range(-1,2,2):
            neighbors.append(array[(x[0]+i)%array.shape[0], x[1]])
        for j in range(-1,2,2):
            neighbors.append(array[x[0], (x[1]+j)%array.shape[1]])
        return np.array(neighbors)


    def energy_difference(x, array):
        dE = 2 * array[x] * np.sum(find_neighbors(x, array))
        return dE


    def monte_carlo_step(lattice):
        # choose a random position on lattice
        x = (np.random.randint(lattice.shape[0]), np.random.randint(lattice.shape[1]))

        # calculate energy difference
        dE = energy_difference(x, lattice)

        # If energy difference negative, accept change.  Otherwise, accept with transition probability.  
        if dE <= 0:
            lattice[x] = -1 * lattice[x]
            magnetization[0] = magnetization[0] + 2 * lattice[x]
        else:
            acceptance_prob = np.exp(-dE/job.sp.kT)
            if np.random.rand() < acceptance_prob:
                lattice[x] = -1 * lattice[x]
                magnetization[0] = magnetization[0] + 2 * lattice[x]


    def run_sim(num_steps, lattice):
        for step in range(num_steps):
            if step%job.sp.log_period == 0:
                M_over_time.append(magnetization[0])
            monte_carlo_step(lattice)


    run_sim(2*job.sp.init_step, Lambda)


    with open(job.fn('init.npy'), 'wb') as f:
        np.save(f, Lambda)
    with open(job.fn('init_magnetizations.npy'), 'wb') as f:
        np.save(f, M_over_time)



@Project.label
def dumped(job):
    return job.isfile(job.fn('lambda.npy'))


# @Project.operation.with_directives({'walltime': 1, 'nranks': 1})
@Project.operation
@Project.pre.after(initialize)
@Project.post(dumped)
def equilibrate(job):
    
    with open(job.fn('init.npy'), 'rb') as f:
        Lambda = np.load(f)
    

    M_over_time = []

    configurations = []
    magnetization = [np.sum(Lambda)]

    
    def find_neighbors(x, array):
        neighbors = []
        for i in range(-1,2,2):
            neighbors.append(array[(x[0]+i)%array.shape[0], x[1]])
        for j in range(-1,2,2):
            neighbors.append(array[x[0], (x[1]+j)%array.shape[1]])
        return np.array(neighbors)


    def energy_difference(x, array):
        dE = 2 * array[x] * np.sum(find_neighbors(x, array))
        return dE


    def monte_carlo_step(lattice):
        # choose a random position on lattice
        x = (np.random.randint(lattice.shape[0]), np.random.randint(lattice.shape[1]))

        # calculate energy difference
        dE = energy_difference(x, lattice)

        # If energy difference negative, accept change.  Otherwise, accept with transition probability.  
        if dE <= 0:
            lattice[x] = -1 * lattice[x]
            magnetization[0] = magnetization[0] + 2 * lattice[x]
        else:
            acceptance_prob = np.exp(-dE/job.sp.kT)
            if np.random.rand() < acceptance_prob:
                lattice[x] = -1 * lattice[x]
                magnetization[0] = magnetization[0] + 2 * lattice[x]


    def run_sim(num_steps, lattice):
        for step in range(num_steps):
            if step%job.sp.log_period == 0:
                M_over_time.append(magnetization[0])
                configurations.append(Lambda.copy())
            monte_carlo_step(lattice)


    run_sim(job.sp.run_step, Lambda)


    with open(job.fn('lambda.npy'), 'wb') as f:
        np.save(f, Lambda)
    with open(job.fn('configurations.npy'), 'wb') as f:
        np.save(f, configurations)
    with open(job.fn('magnetizations.npy'), 'wb') as f:
        np.save(f, M_over_time)



@Project.label
def equil_plotted(job):
    return job.isfile(job.fn('equil_mag.png'))

# @Project.operation.with_directives({'walltime': 1, 'nranks': 1})
@Project.operation
@Project.pre.after(equilibrate)
@Project.post(equil_plotted)
def equil_plot(job):
    import matplotlib.pyplot as plt

    with open(job.fn('lambda.npy'), 'rb') as f:
        Lambda = np.load(f)
    with open(job.fn('magnetizations.npy'), 'rb') as f:
        M_over_time = np.load(f)

    plt.plot(np.arange(len(M_over_time)), np.array(M_over_time)/job.sp.L**2)
    plt.savefig(job.fn("equil_mag.png"))

    plt.figure(figsize=(20,20), dpi= 500)
    plt.matshow(Lambda)
    plt.savefig(job.fn("equil_matrix.png"))



if __name__ == '__main__':
    Project().main()
