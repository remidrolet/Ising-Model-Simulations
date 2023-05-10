import signac
import numpy as np
import matplotlib.pyplot as plt
import sys
import timeit

project = signac.get_project()


def plot_magnetizations():
    temperatures = []
    magnetizations = {}
    for job in project.find_jobs():
        with open(job.fn('magnetizations.npy'), 'rb') as f:
            M_over_time = np.load(f)
        mean = np.mean(M_over_time)
        stdev = np.std(M_over_time)
        magnetizations[job.sp.kT] = [mean, stdev]
        temperatures.append(job.sp.kT)

    sorted_temps = sorted(temperatures)
    sorted_mags = np.array([magnetizations[temp] for temp in sorted(temperatures)])
    sorted_mags = sorted_mags/200**2
    plt.errorbar(sorted_temps, sorted_mags[:, 0], yerr=sorted_mags[:, 1], capsize=2.0)
    plt.xlabel("T")
    plt.ylabel("magnetization")
    # plt.title("Magnetization by T")
    plt.savefig("magnetization_by_T_labelled.png")


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

def plot_two_point():
    temperatures = [2.25, 2.3, 2.35]
    for temperature in temperatures:
        for job in project.find_jobs():
            if job.sp.kT == temperature:
                with open(job.fn('configurations.npy'), 'rb') as f:
                    configurations = np.load(f)
                configurations = list(configurations)

                radii = np.arange(int(configurations[0].shape[0]/2 + 1))
                two_pt_fns = []
                configuration_ids = np.arange(0,500,166)

                for radius in radii:
                    two_pts_r = []
                    for i in range(len(configurations)):
                        if i in configuration_ids:
                            two_pts_r.append(two_point_disconnected_4pts(radius, configurations[i]))
                    mean = np.mean(two_pts_r)
                    stdev = np.std(two_pts_r)
                    two_pt_fns.append([mean, stdev])

                with open(job.fn("two_pt_fns.npy"), "wb") as f:
                    np.save(f, two_pt_fns)

                plt.errorbar(radii, np.array(two_pt_fns)[:, 0], yerr=np.array(two_pt_fns)[:, 1])
                plt.savefig(job.fn(f"two_point_correlator_{temperature}.png"))



def plot_from_file():
    temperatures = [2.25, 2.3, 2.35]
    for temperature in temperatures:
        for job in project.find_jobs():
            if job.sp.kT == temperature:
                with open(job.fn('two_pt_fns.npy'), 'rb') as f:
                    two_pt_fns = np.load(f)
                with open(job.fn('configurations.npy'), 'rb') as f:
                    configurations = np.load(f)
                configurations = list(configurations)

                radii = np.arange(int(configurations[0].shape[0]/2 + 1))

                # Create plots
                fig, ax = plt.subplots()
                ax.errorbar(radii, np.array(two_pt_fns)[:, 0], yerr=np.array(two_pt_fns)[:, 1], capsize=2.0)
                ax.plot(radii, [radius**(-1/4) for radius in radii], color="#1f77b4", linestyle="dashed")
                ax.set_xlabel("r")
                ax.set_ylabel("f(r)")
                ax.set_title(f"Two-point Correlation Function")
                plt.savefig(f"two_point_correlator_w_theory_{temperature}.png")


def plot_log_from_file():
    temperatures = [2.25, 2.3, 2.35]
    for temperature in temperatures:
        for job in project.find_jobs():
            if job.sp.kT == temperature:
                with open(job.fn('two_pt_fns.npy'), 'rb') as f:
                    two_pt_fns = np.load(f)
                with open(job.fn('configurations.npy'), 'rb') as f:
                    configurations = np.load(f)
                configurations = list(configurations)

                radii = np.arange(int(configurations[0].shape[0]/2 + 1))

                # Create plots
                fig, ax = plt.subplots()
                ax.plot(np.log(radii), np.log(np.array(two_pt_fns)[:, 0]))
                ax.plot(np.log(radii), np.log(np.array([radius**(-1/4) for radius in radii])), color="#1f77b4", linestyle="dashed")
                ax.set_xlabel("r")
                ax.set_ylabel("f(r)")
                ax.set_title(f"Log of Two-point Correlation Function")
                plt.savefig(f"log_two_point_correlator_w_theory_{temperature}.png")



if __name__ == "__main__":
    globals()[sys.argv[1]]()