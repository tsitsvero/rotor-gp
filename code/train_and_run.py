#!/usr/bin/env python
# coding: utf-8

# In[2]:


# ! jupyter nbconvert --to python --RegexRemovePreprocessor.patterns="^%"  train_and_run.ipynb

# RULES:
# 1. Do not make plot.show() only plot.savefig()


# ## Data loading part

# In[85]:


# import argparse

# # Construct the argument parser
# ap = argparse.ArgumentParser()

# # Add the arguments to the parser
# ap.add_argument("-a", "--foperand", required=True,
#    help="first operand")
# ap.add_argument("-b", "--soperand", required=True,
#    help="second operand")
# args = vars(ap.parse_args())

# print(args)

# print("Starting script...")
import time
start_time = time.time()

# export R=0; python ./train_and_run.py 2>&1 

import ase
from ase import io
import os
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append("../../fande") 
# sys.path.append("..") 
import fande

from datetime import datetime

os.environ['VASP_PP_PATH'] = "/home/qklmn/repos/pseudos"

def make_calc_dir():
    if os.getcwd()[-4:] != 'code':
        os.chdir('../../../code')
    dir_name = f'{datetime.now().strftime("%Y-%m-%d_%H %M-%S_%f")}'
    dir_name = '../results/train_and_run/' + dir_name
    os.makedirs(dir_name, exist_ok=True)
    abs_dir_path = os.path.abspath(dir_name)
    return abs_dir_path

temp_dir = make_calc_dir()
os.chdir(temp_dir)
print("Saving data to directory: ", temp_dir)


import sys
log_file = temp_dir + '/LOG_GPU.log'
sys.stdout = open(log_file, 'w')

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

data_folder = "/home/qklmn/data/"
# data_folder = "/data1/simulations/"

print("Loading training data...")
# traj_300 = io.read("/data1/simulations/datasets/rotors/different_temperatures/300/OUTCAR", format="vasp-out", index = ":")
# traj_600 = io.read("/data1/simulations/datasets/rotors/different_temperatures/600/OUTCAR", format="vasp-out", index = ":")
# traj_900 = io.read("/data1/simulations/datasets/rotors/different_temperatures/900/OUTCAR", format="vasp-out", index = ":")
# traj_1200 = io.read("/data1/simulations/datasets/rotors/different_temperatures/1200/OUTCAR", format="vasp-out", index = ":")
# traj_1500 = io.read("/data1/simulations/datasets/rotors/different_temperatures/1200/OUTCAR", format="vasp-out", index = ":")
# traj_1800 = io.read("/data1/simulations/datasets/rotors/different_temperatures/1800/OUTCAR", format="vasp-out", index = ":")
# traj_2100 = io.read("/data1/simulations/datasets/rotors/different_temperatures/2100/OUTCAR", format="vasp-out", index = ":")
# print(len(traj_300), len(traj_600), len(traj_900), len(traj_1200), len(traj_1500), len(traj_1800), len(traj_2100))


traj_300 = io.read(data_folder + "datasets/rotors/different_temperatures/300/OUTCAR", format="vasp-out", index = ":")
# traj_600 = io.read(data_folder + "datasets/rotors/different_temperatures/600/OUTCAR", format="vasp-out", index = ":")
# traj_900 = io.read(data_folder + "datasets/rotors/different_temperatures/900/OUTCAR", format="vasp-out", index = ":")
# traj_1200 = io.read(data_folder + "datasets/rotors/different_temperatures/1200/OUTCAR", format="vasp-out", index = ":")
traj_1500 = io.read(data_folder + "datasets/rotors/different_temperatures/1500/OUTCAR", format="vasp-out", index = ":")
# traj_1800 = io.read(data_folder + "datasets/rotors/different_temperatures/1800/OUTCAR", format="vasp-out", index = ":")
# traj_2100 = io.read(data_folder + "datasets/rotors/different_temperatures/2100/OUTCAR", format="vasp-out", index = ":")

# traj_train = traj_1500[::10].copy()
traj_train = traj_1500[::10].copy()
# training_indices = np.sort(  np.arange(0, 500, 5) )  
# traj_train = [traj_md[i] for i in training_indices]

traj_test = traj_300[300:320].copy()
# test_indices = np.sort(  np.random.choice(np.arange(0,92795), 200, replace=False) ) 
# test_indices = np.sort(  np.arange(400,410,1) ) 
# traj_test = [traj_md[i] for i in test_indices]


import os
machine_name = os.uname()[1]

import wandb


wandb.init(project="rotor-gp", save_code=True, notes="hello", id=machine_name)

## Train data:
energies_train = np.zeros(len(traj_train) )
forces_train = np.zeros( (len(traj_train), len(traj_train[0]), 3 ) )
for i, snap in enumerate(traj_train):
    energies_train[i] = snap.get_potential_energy()
    forces_train[i] = snap.get_forces()
train_data = {'trajectory': traj_train, 'energies': energies_train, 'forces': forces_train}

## Test data:
energies_test = np.zeros(len(traj_test) )
forces_test = np.zeros( (len(traj_test), len(traj_test[0]), 3 ) )
for i, snap in enumerate(traj_test):
    energies_test[i] = snap.get_potential_energy()
    forces_test[i] = snap.get_forces()
test_data = {'trajectory': traj_test, 'energies': energies_test, 'forces': forces_test}


from fande.data import FandeDataModuleASE

atoms = traj_train[0].copy()
H_atoms = [atom.index for atom in atoms if atom.symbol == "H"]
C_atoms = [atom.index for atom in atoms if atom.symbol == "C"]
O_atoms = [atom.index for atom in atoms if atom.symbol == "O"]
N_atoms = [atom.index for atom in atoms if atom.symbol == "N"]
Si_atoms = [atom.index for atom in atoms if atom.symbol == "Si"]
atomic_groups = [H_atoms, C_atoms, O_atoms, N_atoms, Si_atoms]

train_centers_positions = list(range(len(atoms)))
train_derivatives_positions = list(range(len(atoms)))

# Hyperparameters:
hparams = {}

# Descriptors parameters:
soap_params = {
    'species': ["H", "C", "O", "N", "Si"],
    'periodic': True,
    'rcut': 3.0,
    'sigma': 0.5,
    'nmax': 5,
    'lmax': 5,
    'average': "off",
    'crossover': True,
    'dtype': "float64",
    'n_jobs': 10,
    'sparse': False,
    'positions': [7, 11, 15] # ignored
}

fdm = FandeDataModuleASE(train_data, test_data, hparams)

fdm.calculate_invariants_librascal(
    soap_params,
    atomic_groups = atomic_groups,
    centers_positions = train_centers_positions, 
    derivatives_positions = train_derivatives_positions,
    same_centers_derivatives=True,
    frames_per_batch=1,
    calculation_context="train")

fdm.calculate_invariants_librascal(
    soap_params,
    atomic_groups = atomic_groups,
    centers_positions = train_centers_positions, 
    derivatives_positions = train_derivatives_positions,
    same_centers_derivatives=True,
    frames_per_batch=1,
    calculation_context="test")



for g in range(len(atomic_groups)):
    print("\n-----------")
    print("Group ", g)
    print("-----------")
    plt.plot(fdm.train_F[g].cpu()[1::1], linestyle = 'None', marker='o', label='train')
    plt.plot(fdm.test_F[g].cpu()[1::1], linestyle = 'None', marker='x', label='test')
    plt.savefig("tran_test_forces_group_" + str(g) + ".png")
    plt.close()

    plt.hist(fdm.test_F[g].cpu().numpy())
    plt.savefig("histogram_forces_group_" + str(g) + ".png")
    plt.close()
    print(f"Number of training points for group {g}: ", fdm.train_DX[g].shape[-2])
    print(f"Number of test points for group {g}: ", fdm.test_DX[g].shape[-2])



import torch
import numpy as np

total_training_random_samples = 10
high_force_samples = 5

random_samples = total_training_random_samples - high_force_samples


indices_high_force = torch.concat( 
    (torch.topk(fdm.test_F[0], high_force_samples//2, largest=True)[1],  
     torch.topk(fdm.test_F[0], high_force_samples//2, largest=False)[1]) ).cpu().numpy()

ind_slice = np.sort(  
    np.random.choice(np.setdiff1d(np.arange(0, fdm.train_F[0].shape[0]), indices_high_force), random_samples, replace=False) )

indices = np.concatenate((ind_slice, indices_high_force))
indices = np.unique(indices)


print("High force indices: ", indices_high_force)
print(ind_slice)


# ## Training part

# In[ ]:





# In[53]:


# torch.count_nonzero(fdm.train_DX[2][100])

# fdm.train_DX[1].shape


# ## Testing part

# In[87]:


### TESTING PREDICITONS ###
print('Testing performance with (meta-)dynamics run...')

from fande.predict import PredictorASE

model_e = None
trainer_e = None

AG_force_model.eval()

predictor = PredictorASE(
            fdm,
            model_e,
            trainer_e,
            AG_force_model,
            # trainer_f,
            hparams,
            soap_params
)

predictor.test_errors(view_worst_atoms=True)


# In[72]:


### MD with fande calc
from fande.ase import FandeCalc
from ase.units import Bohr,Rydberg,kJ,kB,fs,Hartree,mol,kcal


# from ase.geometry.analysis import Analysis
# from ase.constraints import FixAtoms, FixBondLengths
from ase.optimize import BFGS
from ase import units
from ase.io import read
import logging
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.md.langevin import Langevin
from ase.md.npt import NPT
from ase.md.nptberendsen import NPTBerendsen
from ase.md.nvtberendsen import NVTBerendsen


logging.getLogger("pytorch_lightning").setLevel(logging.ERROR) # logging.ERROR to disable or INFO

# traj_md = read('../results/test/machine_learning/dftb_opt_1000_six_rings.traj', index=":")
# traj_opt = read('../results/test/machine_learning/opt.traj', index=":")

# atoms = fdm.mol_traj[10].copy()
# atoms = traj_md[300].copy()
# atoms = traj_opt[-1].copy()
atoms = traj_test[10].copy()
atoms.set_pbc(True)


atoms.calc = FandeCalc(predictor)
# atoms.calc.set_atomic_groups([rings_carbons, rings_hydrogens], titles=["Rings carbons", "Rings hydrogens"])
# atoms.calc.set_forces_errors_plot_file("../results/test/md_runs/forces_errors.png", loginterval=1)
# atoms.calc = LennardJones()

# Verlet dynamics:
MaxwellBoltzmannDistribution(atoms, temperature_K=300)
# dyn = VelocityVerlet(
#     atoms,
#     dt = 0.5*units.fs,
#     trajectory="../results/test/md_runs/md_test.traj",
#     logfile="../results/test/md_runs/md_log.log",
# )

# dyn = NPT(
#     atoms,
#     # dt = 0.5*units.fs,
#     timestep=0.1,
#     temperature_K=300,
#     externalstress=0.0,
#     trajectory="../results/test/md_runs/md_test.traj",
#     logfile="../results/test/md_runs/md_log.log",
# )

# dyn = NPTBerendsen(atoms, timestep=0.1 * units.fs, temperature_K=300,
#                    taut=100 * units.fs, pressure_au=1.01325 * units.bar,
#                    taup=1000 * units.fs, compressibility=4.57e-5 / units.bar,
#                    trajectory="../results/test/md_runs/md_test.traj",
#                    logfile="../results/test/md_runs/md_log.log",)

# import os

os.makedirs("md_run/", exist_ok=True)

# dyn = NVTBerendsen(atoms, 0.5 * units.fs, 300, taut=0.5*1000*units.fs, 
#                    trajectory="md_run/md_test.traj",   
#                    logfile="md_run/md_log.log")

# dyn.run(100)

# Langevin dynamics:
# https://databases.fysik.dtu.dk/ase/tutorials/md/md.html
MaxwellBoltzmannDistribution(atoms, temperature_K=300)
dyn = Langevin(atoms, 0.1*fs, temperature_K=0.1/units.kB, friction=0.1,
               fixcm=True, trajectory='md_run/md_test.traj',
               logfile="md_run/md_log.log")
dyn.run(500)

# # Structure optimization:
# dyn = BFGS(
#     atoms,
#     trajectory="../results/test/md_runs/md_test.traj",
#     logfile="../results/test/md_runs/md_log.log",)
# dyn.run(fmax=0.1)


print(" ALL JOBS WITHIN PYTHON SCRIPT ARE DONE! ")

print("TIMING: ", time.time()-start_time, " seconds")


# In[67]:


from ase.units import Bohr,Rydberg,kJ,kB,fs,Hartree,mol,kcal


# In[71]:


0.1/fs

