import signac
import numpy as np
import random

project = signac.init_project("micro-panel-project")

T = [2.2, 2.25, 2.35, 2.5]#[0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.1, 2.3, 2.4, 2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0]
L = 200

for kT in T:
    sp = {
            # System Information
            'seed': random.randint(1, 100000),
            'replica': 0,
            # Simulation Setup
            'kT': kT,
            'L': L,
            'run_step': int(5e6),
            'init_step': int(1e6),
            'log_period': int(1e4)
            }
    job = project.open_job(sp)
    job.init()
