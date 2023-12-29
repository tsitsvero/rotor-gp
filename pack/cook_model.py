import os
DATA_DIR = os.path.expanduser("~/repos/data/")
# DATA_DIR = "/data1/simulations/datasets/rotors/high_temp_ML_training_data/"
RESULTS_DIR = os.path.expanduser("~/repos/data/results")

# DATA_DIR = os.path.expanduser("/content/drive/MyDrive/data/")
# # FANDE_DIR = os.path.expanduser("~/")
# RESULTS_DIR = os.path.expanduser("~/results")
# os.makedirs(RESULTS_DIR, exist_ok=True)

ENERGY_MODEL = 'variational_inducing_points' #'variational_inducing_points', 'exact'
ENERGY_LR = 0.01
ENERGY_NUM_STEPS = 5

FORCES_MODEL = 'variational_inducing_points' #'variational_inducing_points', 'exact'
NUM_FORCE_SAMPLES = 10
FORCES_LR = 0.01
FORCES_NUM_STEPS = 5



import argparse
# Create the parser
parser = argparse.ArgumentParser()
# Add an argument
parser.add_argument('--data_dir', type=str, required=True)
parser.add_argument('--results_dir', type=str, required=True)

parser.add_argument('--energy_model', type=str, required=True)
parser.add_argument('--energy_num_inducing_points', type=int, required=True)
parser.add_argument('--energy_lr', type=float, required=True)
parser.add_argument('--energy_num_steps', type=int, required=True)



parser.add_argument('--forces_model', type=str, required=True)
parser.add_argument('--forces_num_inducing_points', type=int, required=True)
parser.add_argument('--num_force_samples', type=int, required=True)
parser.add_argument('--forces_lr', type=float, required=True)
parser.add_argument('--forces_num_steps', type=int, required=True)

parser.add_argument('--predictor_name', type=str, required=True)
parser.add_argument('--subsample', type=int, required=False, default=200)


# Parse the argument
args = parser.parse_args()
DATA_DIR = args.data_dir
RESULTS_DIR = args.results_dir
ENERGY_MODEL = args.energy_model
ENERGY_NUM_INDUCING_POINTS = args.energy_num_inducing_points
ENERGY_LR = args.energy_lr
ENERGY_NUM_STEPS = args.energy_num_steps

FORCES_MODEL = args.forces_model
FORCES_NUM_INDUCING_POINTS = args.forces_num_inducing_points
NUM_FORCE_SAMPLES = args.num_force_samples
FORCES_LR = args.forces_lr
FORCES_NUM_STEPS = args.forces_num_steps

PREDICTOR_NAME = args.predictor_name
# PREDICTOR_NAME = "fande_predictor_" + str(ENERGY_MODEL) + "_" + str(ENERGY_LR) + "_" + str(ENERGY_NUM_STEPS) + "_" + str(FORCES_MODEL) + "_" + str(NUM_FORCE_SAMPLES) + "_" + str(FORCES_LR) + "_" + str(FORCES_NUM_STEPS) + ".pth"

SUBSAMPLE = args.subsample

print("DATA_DIR", DATA_DIR)
print("RESULTS_DIR", RESULTS_DIR)
print("ENERGY_MODEL", ENERGY_MODEL)
print("ENERGY_NUM_INDUCING_POINTS", ENERGY_NUM_INDUCING_POINTS)
print("ENERGY_LR", ENERGY_LR)
print("ENERGY_NUM_STEPS", ENERGY_NUM_STEPS)
print("FORCES_MODEL", FORCES_MODEL)
print("FORCES_NUM_INDUCING_POINTS", FORCES_NUM_INDUCING_POINTS)
print("NUM_FORCE_SAMPLES", NUM_FORCE_SAMPLES)
print("FORCES_LR", FORCES_LR)
print("FORCES_NUM_STEPS", FORCES_NUM_STEPS)
print("PREDICTOR_NAME", PREDICTOR_NAME)

os.makedirs(RESULTS_DIR, exist_ok=True)

FANDE_DIR = os.path.expanduser("~/repos/")

############################################################################################################
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

trajectory_forces = traj_295_2000K[0:5000:5]
trajectory_forces = trajectory_forces[::SUBSAMPLE].copy()
# trajectory_energy = traj_295[0:5000] + traj_355[0:5000] + traj_295_2000K[0:5000] + traj_355_2000K[0:5000] + traj_295_2000K_forced[0:5000] + traj_355_2000K_forced[0:5000]
# trajectory_energy = traj_295 + traj_295_2000K + traj
trajectory_energy = traj_295[0:5000:10] +  traj_295_2000K[0:5000:10]
trajectory_energy = trajectory_energy[::SUBSAMPLE].copy()
print(len(trajectory_forces), len(trajectory_energy))



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

total_forces_samples_per_group = [NUM_FORCE_SAMPLES] * len(atomic_groups)
high_forces_samples_per_group = [0] * len(atomic_groups)
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
# Making energy model

from fande.models import EnergyModel

hparams = {
        'dtype' : 'float32',
        # 'device' : 'gpu',
        'device' : 'cpu',
        'energy_model_hparams' : {
                'model_type' : ENERGY_MODEL,#'variational_inducing_points', 'exact'
                'num_inducing_points' : ENERGY_NUM_INDUCING_POINTS,
                'num_epochs' : ENERGY_NUM_STEPS,
                'learning_rate' : ENERGY_LR,
        }
        }

Energy_model = EnergyModel(
        dataloader_energy,
        hparams=hparams)

Energy_model.fit()


print("Energy model fitted")
# # Fitting forces

from fande.models import ModelForces, GroupModelForces


n_steps_list = [FORCES_NUM_STEPS] * len(atomic_groups)
lr_list = [FORCES_LR] * len(atomic_groups)

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
        'forces_model_hparams' : {
                'model_type' : FORCES_MODEL,#'variational_inducing_points', 'exact'
                'num_inducing_points' : FORCES_NUM_INDUCING_POINTS,
        }
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
fande_calc.save_predictor(RESULTS_DIR + "/" + PREDICTOR_NAME)
# device = torch.device('cuda:0') # always specify the gpu id!
device = torch.device('cpu')
fande_calc.predictor.move_models_to_device(device)
from ase import io
from tqdm import tqdm
test_traj = io.read(DATA_DIR + "/results_triasine_ML_2000/struct_295_295K/md_trajectory.traj", index="1235:1240")
test_traj = test_traj.copy()

real_energies = [s.get_potential_energy() for s in test_traj]
predicted_energies = []
for i in tqdm(range(len(test_traj))):
        test_traj[i].calc = fande_calc
        # predicted_energies.append( test_traj[i].get_potential_energy() )
        # print(test_traj[i].get_potential_energy() )
        print(test_traj[i].get_forces())
        print(test_traj[i].calc.get_forces_variance(test_traj[i]))
        print(test_traj[i].get_potential_energy() )
        # test_traj[i].get_forces()