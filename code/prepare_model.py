#!/usr/bin/env python


import numpy as np
import torch
from ase import io

traj_295 = io.read("/data1/simulations/datasets/rotors/high_temp_ML_training_data/results_triasine_ML_2000/struct_295_295K/md_trajectory.traj", index=":")
traj_355 = io.read("/data1/simulations/datasets/rotors/high_temp_ML_training_data/results_triasine_ML_2000/struct_355_355K/md_trajectory.traj", index=":")

traj_295_2000K = io.read("/data1/simulations/datasets/rotors/high_temp_ML_training_data/results_triasine_ML_2000/struct_295_2000K/md_trajectory.traj", index=":")
traj_355_2000K = io.read("/data1/simulations/datasets/rotors/high_temp_ML_training_data/results_triasine_ML_2000/struct_355_2000K/md_trajectory.traj", index=":")
traj_295_2000K_forced = io.read("/data1/simulations/datasets/rotors/high_temp_ML_training_data/results_triasine_ML_2000/struct_295_2000K_0075force/md_trajectory.traj", index=":")
traj_355_2000K_forced = io.read("/data1/simulations/datasets/rotors/high_temp_ML_training_data/results_triasine_ML_2000/struct_355_2000K_0075force/md_trajectory.traj", index=":")



trajectory_forces = traj_295_2000K[0:5000:10]
trajectory_forces = trajectory_forces[0::10].copy()
trajectory_energy = traj_295[0:5000] + traj_355[0:5000] + traj_295_2000K[0:5000] + traj_355_2000K[0:5000] + traj_295_2000K_forced[0:5000] + traj_355_2000K_forced[0:5000]
# trajectory_energy = traj_295 + traj_295_2000K + traj
trajectory_energy = trajectory_energy[::10].copy()
print("Length of trajectory forces and energy:", len(trajectory_forces), len(trajectory_energy))