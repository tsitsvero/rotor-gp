# %%
import os
DATA_DIR = os.path.expanduser("~/repos/data/")
# DATA_DIR = "/data1/simulations/datasets/rotors/high_temp_ML_training_data/"

FANDE_DIR = os.path.expanduser("~/repos/")
RESULTS_DIR = os.path.expanduser("~/repos/data/results")
os.makedirs(RESULTS_DIR, exist_ok=True)

import os
import sys
import torch
sys.path.append(FANDE_DIR + "fande/")


from ase import io


traj_295 = io.read(DATA_DIR+"/results_triasine_ML_2000/struct_295_295K/md_trajectory.traj", index=":")
# traj_355 = io.read(DATA_DIR+"/results_triasine_ML_2000/struct_355_355K/md_trajectory.traj", index=":")
traj_295_2000K = io.read(DATA_DIR+"/results_triasine_ML_2000/struct_295_2000K/md_trajectory.traj", index=":")
# traj_355_2000K = io.read(DATA_DIR+"/results_triasine_ML_2000/struct_355_2000K/md_trajectory.traj", index=":")
# traj_295_2000K_forced = io.read(DATA_DIR+"/results_triasine_ML_2000/struct_295_2000K_0075force/md_trajectory.traj", index=":")
# traj_355_2000K_forced = io.read(DATA_DIR+"/results_triasine_ML_2000/struct_355_2000K_0075force/md_trajectory.traj", index=":")


# %%
trajectory_forces = traj_295_2000K[0:5000:5]
trajectory_forces = trajectory_forces[:].copy()

# trajectory_energy = traj_295[0:5000] + traj_355[0:5000] + traj_295_2000K[0:5000] + traj_355_2000K[0:5000] + traj_295_2000K_forced[0:5000] + traj_355_2000K_forced[0:5000]
# trajectory_energy = traj_295 + traj_295_2000K + traj
trajectory_energy = traj_295[0:5000:10] +  traj_295_2000K[0:5000:10]
trajectory_energy = trajectory_energy[:].copy()

print(len(trajectory_forces), len(trajectory_energy))

# %%
from fande.data import FandeDataModule
from fande.utils.find_atomic_groups import find_atomic_groups


soap_params = dict(soap_type="PowerSpectrum",
        interaction_cutoff=4.0,
        max_radial=4,
        max_angular=4,
        gaussian_sigma_constant=0.3,
        gaussian_sigma_type="Constant",
        cutoff_function_type="RadialScaling",
        cutoff_smooth_width=0.1, # 0.1 is way better than 0.5
        cutoff_function_parameters=
                dict(
                        rate=1,
                        scale=3.5,
                        exponent=4
                        ),
        radial_basis="GTO",
        normalize=True, # setting False makes model untrainable
        #   optimization=
        #         dict(
        #                 Spline=dict(
        #                    accuracy=1.0e-05
        #                 )
        #             ),
        compute_gradients=True, # for energies gradients are ignored
        expansion_by_species_method='structure wise'
        )
##FOR NOW USE THE SAME SOAP PARAMETERS FOR ENERGY AND FORCES! (that makes sense if you're modeling the MD)

sample_snapshot = trajectory_forces[0].copy()
fdm = FandeDataModule()
atomic_groups = find_atomic_groups(sample_snapshot)
train_centers_positions = sum(atomic_groups, []) #list(range(len(atoms)))
train_derivatives_positions = sum(atomic_groups, [])#list(range(len(atoms)))
fdm.atomic_groups_sample_snapshot = sample_snapshot.copy()
fdm.atomic_groups = atomic_groups

total_forces_samples_per_group = [3000] * len(atomic_groups)
high_forces_samples_per_group = [0] * len(atomic_groups)

# %%
dataloader_energy, dataloaders_forces = fdm.dataloaders_from_trajectory(
                                                                trajectory_energy,
                                                                trajectory_forces,
                                                                # energies = None,
                                                                # forces = None,
                                                                atomic_groups = atomic_groups,
                                                                centers_positions = train_centers_positions,
                                                                derivatives_positions = train_derivatives_positions,
                                                                energy_soap_hypers = soap_params,
                                                                forces_soap_hypers = soap_params,
                                                                total_forces_samples_per_group = total_forces_samples_per_group,
                                                                high_force_samples_per_group = high_forces_samples_per_group,
                                                                )

# %%
# Making energy model

from fande.models import EnergyModel

hparams = {
        'dtype' : 'float32',
        # 'device' : 'gpu',
        'device' : 'cpu',
        'energy_model_hparams' : {
                'model_type' : 'exact',#'variational_inducing_points', 'exact'
                'num_inducing_points' : 10,
                'num_epochs' : 5_00,
                'learning_rate' : 0.01,
        }
        }

Energy_model = EnergyModel(
        dataloader_energy,
        hparams=hparams)

Energy_model.fit()

# %%
# # Fitting forces

from fande.models import ModelForces, GroupModelForces


n_steps_list = [5_00] * len(atomic_groups)
lr_list = [0.01] * len(atomic_groups)

models_hparams = []
for i in range(len(atomic_groups)):
        model_hparams = {
        'atomic_group' : atomic_groups[i],
        'dtype' : hparams['dtype'],
        'device' : hparams['device'],
        'num_epochs' : n_steps_list[i],
        'learning_rate' : lr_list[i],
        'soap_dim' : dataloaders_forces[i].dataset[0][0].shape[-1],
        'soap_params' : soap_params,
        }
        models_hparams.append(model_hparams)

hparams['per_model_hparams'] = models_hparams # access per_model_hparams by model.model_id
gpu_id = 0


models_forces = []
for i in range(len(atomic_groups)):
        model = ModelForces(
        train_x = dataloaders_forces[i].dataset[:][0],
        train_y = dataloaders_forces[i].dataset[:][1],
        atomic_group = atomic_groups[i],
        hparams = hparams,
        id=i)
        models_forces.append(model)

AG_force_model = GroupModelForces(
        models= models_forces,
        train_data_loaders = dataloaders_forces,
        hparams=hparams,
        gpu_id=gpu_id)

AG_force_model.fit()

# %%
from fande.predict import FandePredictor
from fande.ase import FandeCalc

# Energy_model = None
# AG_force_model = None
predictor = FandePredictor(
        fdm,
        AG_force_model,
        Energy_model,
        hparams,
        soap_params
        )

fande_calc = FandeCalc(predictor)
from datetime import datetime
now_str = str( datetime.now() )
device = torch.device('cpu')
fande_calc.predictor.move_models_to_device(device)
fande_calc.save_predictor(RESULTS_DIR + "/fande_predictor_" + now_str + ".pth")

# # %%

# from fande.predict import FandePredictor
# from fande.ase import FandeCalc
# # load the predictor:
# predictor_loaded = torch.load(RESULTS_DIR + "/fande_predictor.pth")
# fande_calc_loaded = FandeCalc(predictor_loaded)
# device = torch.device('cpu')
# fande_calc_loaded.predictor.move_models_to_device(device)

# %%
# device = torch.device('cuda:0') # always specify the gpu id!
# device = torch.device('cpu')
# fande_calc.predictor.move_models_to_device(device)

# # %%
# from ase import io
# from tqdm import tqdm
# test_traj = io.read(DATA_DIR + "/results_triasine_ML_2000/struct_295_295K/md_trajectory.traj", index="1235:1240")
# test_traj = test_traj.copy()

# real_energies = [s.get_potential_energy() for s in test_traj]
# predicted_energies = []
# for i in tqdm(range(len(test_traj))):
#         test_traj[i].calc = fande_calc
#         # predicted_energies.append( test_traj[i].get_potential_energy() )
#         # print(test_traj[i].get_potential_energy() )
#         print(test_traj[i].get_forces())
#         print(test_traj[i].get_potential_energy() )
#         # test_traj[i].get_forces()

# # %%
# test_traj[0].calc.get_forces_variance(test_traj[0])

# # %%
# %%time
# # test_traj[0].get_forces()
# for i in tqdm(range(len(test_traj))):
#         test_traj[i].get_forces()
# # test_traj[1].get_potential_energy()

# # %%
# import matplotlib.pyplot as plt

# plt.plot(real_energies, label="real")
# plt.plot(predicted_energies, label="predicted")
# plt.legend()
# plt.show()

# # %%
# atoms = trajectory_energy[51].copy()

# atoms.set_calculator(fande_calc)

# print(atoms.get_potential_energy())
# print(atoms.get_forces())

# # %% [markdown]
# # ## Testing area

# # %%
# from ase import io
# test_traj = io.read("/data1/simulations/datasets/rotors/high_temp_ML_training_data/results_triasine_ML_2000/struct_295_295K/md_trajectory.traj", index="1000:1100")
# # test_traj = io.read("/data1/simulations/datasets/rotors/high_temp_ML_training_data/results_triasine_ML_2000/struct_295_2000K_0075force/md_trajectory.traj", index="-100:")
# # test_traj = io.read("/data1/simulations/datasets/rotors/high_temp_ML_training_data/295_0.075_same_+/md_trajectory.traj", index="100:150")

# # atoms.set_calculator(fande_calc_loaded)

# # %%
# %%capture c
# from tqdm import tqdm

# energies_true = []
# energies_pred = []

# for i in tqdm(range(len(test_traj))):
#         atoms = test_traj[i].copy()
#         atoms.calc = fande_calc
#         energies_pred.append(atoms.get_potential_energy())
#         energies_true.append(test_traj[i].get_potential_energy())


# # %%
# import matplotlib.pyplot as plt
# plt.plot(energies_true, label="true")
# plt.plot(energies_pred, label="pred")
# plt.legend()
# plt.show()

# # %%
# fande_calc.predictor.energy_model.model.model.variational_strategy.inducing_points

# # import matplotlib.pyplot as plt

# train_x = fande_calc.predictor.energy_model.model.train_x[:].cpu().detach().numpy()
# mean_train_x = train_x.mean(axis=0)

# train_x_no_mean = train_x - mean_train_x
# train_x_variance = train_x_no_mean.var(axis=0)

# plt.figure(figsize=(5,3))
# plt.plot(train_x_variance, color="black")
# plt.xlabel("SOAP feature index")
# plt.ylabel("Variance")
# plt.tight_layout()
# plt.yscale('log')
# # plt.xlim(210,215)
# # plt.ylim(0,2.0)
# # plt.savefig("variance.pdf")
# plt.show()


# # plt.figure(figsize=(5,3))
# # plt.hist(train_x_variance, bins=100, color="black")
# # plt.xlabel("Variance")
# # plt.ylabel("Count")
# # plt.tight_layout()
# # plt.yscale('log')
# # # plt.savefig("variance_hist.pdf")
# # plt.show()

# plt.figure(figsize=(5,3))
# plt.plot(mean_train_x, color="black")
# plt.xlabel("SOAP feature index")
# plt.ylabel("Mean")
# plt.tight_layout()
# # plt.yscale('log')
# # plt.savefig("mean.pdf")
# plt.show()

# # for i in range(0,1000):
# #         # plt.plot(fande_calc.predictor.energy_model.model.model.variational_strategy.inducing_points[i].cpu().detach().numpy().flatten())
# #         plt.plot(train_x[2*i].flatten() - mean_train_x.flatten())
# # # plt.hist(fande_calc.predictor.energy_model.model.train_x.cpu().detach().numpy().flatten(), bins=100)
# # # plt.xlim(50, 60)
# # plt.show()

# # %%
# from ase.build import molecule
# from ase.build import fcc111
# from xtb.ase.calculator import XTB
# from ase import io


# # slab = fcc111('Cu', size=(2,2,3), vacuum=10.0)
# atoms = molecule('H2O')
# # atoms.set_cell([10, 10, 10])
# # atoms.set_pbc(True)
# # io.write("coord.tmol", atoms, format="turbomole")
# # io.write("poscar", atoms, format="vasp")
# # io.write("test.cif", atoms, format="cif")
# # io.write("test.xyz", atoms, format="extxyz")
# # atoms = slab.copy()
# # atoms.calc = XTB(method="GFN2-xTB")
# atoms.calc = XTB(method="gfnff")
# atoms.get_potential_energy()

# atoms.get_forces()


