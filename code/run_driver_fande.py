# To run the server: i-pi input.xml > log &
# To run the driver: python ./run_driver_fande.py --pimd_port 10200 --pimd_num_calcs 0

# i-pi input.xml > log || python ./run_driver.py

# To clean up: 
# rm -rf *log* *PREFIX*  *RESTART* *COLVAR* plumed/*COLVAR* plumed/*HILLS* *HILLS* *colvar* temp_calc_dir_* EXIT

# Tutorial:
# https://gitlab.com/Sucerquia/ase-plumed_tutorial
# https://dftbplus-recipes.readthedocs.io/en/latest/interfaces/ipi/ipi.html

# from ase import Atoms
# class FandeAtomsWrapper(Atoms):   
#     def __init__(self, *args, **kwargs):
#         super(FandeAtomsWrapper, self).__init__(*args, **kwargs)      
#         self.calc_history_counter = 0
#         self.request_variance = False
    
#     def get_forces_variance(self):
#         forces_variance = super(FandeAtomsWrapper, self).calc.get_forces_variance(self)
#         return forces_variance

#     def get_forces(self):       
#         forces = super(FandeAtomsWrapper, self).get_forces()
#         if self.request_variance:
#             forces_variance = super(FandeAtomsWrapper, self).calc.get_forces_variance(self)
#             self.arrays['forces_variance'] = forces_variance
#         # energy = super(AtomsWrapped, self).get_potential_energy()
#         from ase.io import write
#         import os
#         os.makedirs("ase_calc_history" , exist_ok=True)
#         write( "ase_calc_history/" + str(self.calc_history_counter) + ".xyz", self, format="extxyz")
#         # self.calc_history.append(self.copy())       
#         self.calc_history_counter += 1
#         return forces
    
# Import the library
import argparse
# Create the parser
parser = argparse.ArgumentParser()
# Add an argument

parser.add_argument('--pimd_port', type=int, required=True)
parser.add_argument('--pimd_num_calcs', type=int, required=True)

# parser.add_argument('--calc_dir', type=str, required=True)
# parser.add_argument('--vasp_command', type=str, required=True)
# parser.add_argument('--vasp_pseudos_dir', type=str, required=True)
# parser.add_argument('--traj_filename', type=str, required=True)
# parser.add_argument('--log_filename', type=str, required=True)

# parser.add_argument('--temperature', type=float, required=True)
# parser.add_argument('--md_time_step', type=float, required=True)
# parser.add_argument('--num_steps', type=int, required=True)
# parser.add_argument('--friction', type=float, required=True)

# # parser.add_argument('--rotors_list', type=list, required=True)
# parser.add_argument('--forces_alpha', type=float, required=True, nargs="+")

# parser.add_argument('--calculator', type=str, required=False, default="vasp")

# parser.add_argument('--structure', type=str, required=True)

# parser.add_argument('--moving_method', type=str, required=False, default="Langevin")

# parser.add_argument('--relaxation_interval', type=int, required=True)
# parser.add_argument('--rotation_angle', type=float, required=True)
# Parse the argument
args = parser.parse_args()


PIMD_PORT = args.pimd_port
PIMD_NUM_CALCS = args.pimd_num_calcs







import os
import sys

from ase.calculators.socketio import SocketClient, SocketIOCalculator
from ase.io import read,write
from ase import units

# from ase.calculators.lj import LennardJones
# from xtb.ase.calculator import XTB
from ase.calculators.dftb import Dftb



# Define atoms object
# atoms = read("init.xyz", index='0', format='extxyz')
# atoms = read(' ../../../structures/structures/new_systems/ktu_002.xyz')

# atoms = read('/home/qklmn/repos/structures/structures/new_systems/ktu_002.cif')
# atoms = read('/home/qklmn/data/starting_configuration/1.cif') # atoms specified here should be the same as in i-pi input file (otherwise atomic order differ, structure blows up!)
# /home/qklmn/data/starting_configuration/triazine/POSCAR.OTIPS_355K-opt-structure-fixed-lattice-supercell.cif
# atoms = read("/home/qklmn/data/starting_configuration/triazine/POSCAR.OTIPS_355K-opt-structure-fixed-lattice-supercell.cif")

# Set up the calculator #################
# calc_base = XTB(method="GFN2-xTB")
# calc = Plumed(calc=calc_base,
#                     input=setup,
#                     timestep=timestep,
#                     atoms=atoms,
#                     kT=0.1)

import os


# Hyperparameters:
hparams = {
        'dtype' : 'float32',
        'device' : 'gpu'
        }

# Descriptors parameters:
# https://github.com/lab-cosmo/librascal/blob/master/examples/MLIP_example.ipynb
soap_params = {
# 'species': ["H", "C", "O", "N", "Si"],
# 'periodic': True,
'interaction_cutoff': 3.5, #5
'gaussian_sigma_constant': 0.3,
'max_radial': 4, #5
'max_angular': 4,#5
'cutoff_smooth_width': 0.1,
# 'average': "off",
# 'crossover': True,
# 'dtype': "float64",
# 'n_jobs': 10,
# 'sparse': False,
# 'positions': [7, 11, 15] # ignored
}


#### for rotors
from ase import io

STRUCTURE="355K"

# if STRUCTURE == "295K":
#     # crystal = io.read( os.path.expanduser("/home/qklmn/data/starting_configuration/triazine/295optdftb.cif"), format="cif" )
#     crystal = io.read( os.path.expanduser("/home/dlbox2/ダウンロード/artificial-rotor/structures/triazine/ipi/295Ksupercell.cif"), format="cif" ) 
#     # 295 K structure:
#     triazine_1 = [6, 8, 10, 102, 104, 106]
#     triazine_2 = [9, 7, 11, 103, 105, 107]
#     triazine_3 = [270, 272, 274, 366, 368, 370]
#     triazine_4 = [271, 273, 275, 367, 369, 371]
#     axis_ring_1 = [27, 33]
#     ring_1 = [25, 29, 31, 35]
#     ring_1_full = ring_1 + axis_ring_1 + [123, 125, 127, 129]
#     axis_ring_2 = [308, 302]
#     ring_2 = [304, 306, 310, 312]
#     ring_2_full = ring_2 + axis_ring_2 + [396, 398, 400, 402]
#     axis_ring_3 = [282, 276]
#     ring_3 = [278, 280, 284, 286]
#     ring_3_full = ring_3 + axis_ring_3 + [378, 380, 382, 384]
#     axis_ring_4 = [291, 297]
#     ring_4 = [289, 293, 295, 299]
#     ring_4_full = ring_4 + axis_ring_4 + [387, 389, 391, 393]
#     axis_ring_5 = [18, 12]
#     ring_5 = [14, 16, 20, 22]
#     ring_5_full = ring_5 + axis_ring_5 + [114, 116, 118, 120]
#     axis_ring_6 = [44, 38]
#     ring_6 = [40, 42, 46, 48]
#     ring_6_full = ring_6 + axis_ring_6 + [132, 134, 136, 138]

#     axis_ring_A = [290, 296]
#     ring_A = [292, 294, 288, 298]
#     ring_A_full = ring_A + axis_ring_A + [386, 388, 390, 392]
#     axis_ring_B = [309, 303]
#     ring_B = [305, 307, 311, 313]
#     ring_B_full = ring_B + axis_ring_B + [397, 399, 401, 403]
#     axis_ring_C = [283, 277]
#     ring_C = [279, 281, 285, 287]
#     ring_C_full = ring_C + axis_ring_C + [379, 381, 383, 385]

#     axis_ring_D = [26, 32]
#     ring_D = [24, 28, 30, 34]
#     ring_D_full = ring_D + axis_ring_D + [122, 124, 126, 128]

#     axis_ring_E = [45, 39]
#     ring_E = [41, 43, 47, 49]
#     ring_E_full = ring_E + axis_ring_E + [133, 135, 137, 139] # not rotating

#     axis_ring_F = [19, 13]
#     ring_F = [15, 17, 21, 23]
#     ring_F_full = ring_F + axis_ring_F + [115, 117, 119, 121] # not rotating
#     RuntimeError("you was running with STRUCTURE 355K!")

if STRUCTURE == "355K":
    # crystal = io.read( os.path.expanduser("/home/qklmn/data/starting_configuration/triazine/2.cif"), format="cif" )
    crystal = io.read( os.path.expanduser("/home/dlbox2/ダウンロード/artificial-rotor/structures/triazine/ipi/355Ksupercell.cif"), format="cif" ) 
    # 355 K structure:
    triazine_1 = [6, 8, 10, 102, 104, 106]
    triazine_2 = [9, 7, 11, 103, 105, 107]
    triazine_3 = [270, 272, 274, 366, 368, 370]
    triazine_4 = [271, 273, 275, 367, 369, 371]

    axis_ring_1 = [25, 31]
    ring_1 = [27, 29, 33, 35]
    ring_1_full = ring_1 + axis_ring_1 + [123, 125, 127, 129]
    axis_ring_2 = [310, 304]
    ring_2 = [300, 302, 306, 308]
    ring_2_full = ring_2 + axis_ring_2 + [394, 396, 398, 400]
    axis_ring_3 = [276, 282]
    ring_3 = [278, 280, 284, 286]
    ring_3_full = ring_3 + axis_ring_3 + [378, 380, 382, 384]
    axis_ring_4 = [289, 295]
    ring_4 = [291, 293, 297, 299]
    ring_4_full = ring_4 + axis_ring_4 + [387, 389, 391, 393]
    axis_ring_5 = [12, 18]
    ring_5 = [14, 16, 20, 22]
    ring_5_full = ring_5 + axis_ring_5 + [114, 116, 118, 120]
    axis_ring_6 = [46, 40]
    ring_6 = [36, 38, 42, 44]
    ring_6_full = ring_6 + axis_ring_6 + [130, 132, 134, 136]

    axis_ring_A = [288, 294]
    ring_A = [290, 292, 296, 298]
    ring_A_full = ring_A + axis_ring_A + [386, 388, 390, 392]
    axis_ring_B = [277, 283]
    ring_B = [279, 281, 285, 287]
    ring_B_full = ring_B + axis_ring_B + [379, 381, 383, 385]
    axis_ring_C = [311, 305]
    ring_C = [301, 303, 307, 309]
    ring_C_full = ring_C + axis_ring_C + [395, 397, 399, 401]
    axis_ring_D = [24, 30]
    ring_D = [26, 28, 32, 34]
    ring_D_full = ring_D + axis_ring_D + [122, 124, 126, 128]
    axis_ring_E = [13, 19]
    ring_E = [15, 17, 21, 23]
    ring_E_full = ring_E + axis_ring_E + [115, 117, 119, 121]
    axis_ring_F = [47, 41]
    ring_F = [37, 39, 43, 45]
    ring_F_full = ring_F + axis_ring_F + [131, 133, 135, 137]
else:
    RuntimeError("STRUCTURE must be among 295K and 355K")

atoms = crystal.copy()


import numpy as np
from icecream import ic

from ase import Atoms
import os
class RotationAtomsWrapper(Atoms):   
    def __init__(self, *args, **kwargs):
        super(RotationAtomsWrapper, self).__init__(*args, **kwargs)      
        self.calc_history_counter = 0
        self.forces_alpha = [0.5, -0.5, -0.5, 0.5, -0.5, -0.5,   -0.5, 0.5, 0.5,  -0.5, 0.5, 0.5]    #[0.05] * 12

    def get_forces(self, md=False):       
        forces = super(RotationAtomsWrapper, self).get_forces(md=md)
        # energy = super(AtomsWrapped, self).get_potential_energy()
        # os.makedirs("ase_calc_history" , exist_ok=True)
        # write( "ase_calc_history/" + str(self.calc_history_counter) + ".xyz", self, format="extxyz")
        # self.calc_history.append(self.copy())       
        # self.calc_history_counter += 1

        axis_1_vector = self.positions[axis_ring_1[1]] - self.positions[axis_ring_1[0]]
        axis_1_center = (self.positions[axis_ring_1[1]] + self.positions[axis_ring_1[0]]) / 2.0
        rotor_1_relative_positions = self.positions[ring_1] - axis_1_center
        rotor_1_cross_product = np.cross(rotor_1_relative_positions, axis_1_vector)

        axis_2_vector = self.positions[axis_ring_2[1]] - self.positions[axis_ring_2[0]]
        axis_2_center = (self.positions[axis_ring_2[1]] + self.positions[axis_ring_2[0]]) / 2.0
        rotor_2_relative_positions = self.positions[ring_2] - axis_2_center
        rotor_2_cross_product = np.cross(rotor_2_relative_positions, axis_2_vector)

        axis_3_vector = self.positions[axis_ring_3[1]] - self.positions[axis_ring_3[0]]
        axis_3_center = (self.positions[axis_ring_3[1]] + self.positions[axis_ring_3[0]]) / 2.0
        rotor_3_relative_positions = self.positions[ring_3] - axis_3_center
        rotor_3_cross_product = np.cross(rotor_3_relative_positions, axis_3_vector)

        axis_4_vector = self.positions[axis_ring_4[1]] - self.positions[axis_ring_4[0]]
        axis_4_center = (self.positions[axis_ring_4[1]] + self.positions[axis_ring_4[0]]) / 2.0
        rotor_4_relative_positions = self.positions[ring_4] - axis_4_center
        rotor_4_cross_product = np.cross(rotor_4_relative_positions, axis_4_vector)

        axis_5_vector = self.positions[axis_ring_5[1]] - self.positions[axis_ring_5[0]]
        axis_5_center = (self.positions[axis_ring_5[1]] + self.positions[axis_ring_5[0]]) / 2.0
        rotor_5_relative_positions = self.positions[ring_5] - axis_5_center
        rotor_5_cross_product = np.cross(rotor_5_relative_positions, axis_5_vector)

        axis_6_vector = self.positions[axis_ring_6[1]] - self.positions[axis_ring_6[0]]
        axis_6_center = (self.positions[axis_ring_6[1]] + self.positions[axis_ring_6[0]]) / 2.0
        rotor_6_relative_positions = self.positions[ring_6] - axis_6_center
        rotor_6_cross_product = np.cross(rotor_6_relative_positions, axis_6_vector)

        axis_A_vector = self.positions[axis_ring_A[1]] - self.positions[axis_ring_A[0]]
        axis_A_center = (self.positions[axis_ring_A[1]] + self.positions[axis_ring_A[0]]) / 2.0
        rotor_A_relative_positions = self.positions[ring_A] - axis_A_center
        rotor_A_cross_product = np.cross(rotor_A_relative_positions, axis_A_vector)

        axis_B_vector = self.positions[axis_ring_B[1]] - self.positions[axis_ring_B[0]]
        axis_B_center = (self.positions[axis_ring_B[1]] + self.positions[axis_ring_B[0]]) / 2.0
        rotor_B_relative_positions = self.positions[ring_B] - axis_B_center
        rotor_B_cross_product = np.cross(rotor_B_relative_positions, axis_B_vector)

        axis_C_vector = self.positions[axis_ring_C[1]] - self.positions[axis_ring_C[0]]
        axis_C_center = (self.positions[axis_ring_C[1]] + self.positions[axis_ring_C[0]]) / 2.0
        rotor_C_relative_positions = self.positions[ring_C] - axis_C_center
        rotor_C_cross_product = np.cross(rotor_C_relative_positions, axis_C_vector)

        axis_D_vector = self.positions[axis_ring_D[1]] - self.positions[axis_ring_D[0]]
        axis_D_center = (self.positions[axis_ring_D[1]] + self.positions[axis_ring_D[0]]) / 2.0
        rotor_D_relative_positions = self.positions[ring_D] - axis_D_center
        rotor_D_cross_product = np.cross(rotor_D_relative_positions, axis_D_vector)

        axis_E_vector = self.positions[axis_ring_E[1]] - self.positions[axis_ring_E[0]]
        axis_E_center = (self.positions[axis_ring_E[1]] + self.positions[axis_ring_E[0]]) / 2.0
        rotor_E_relative_positions = self.positions[ring_E] - axis_E_center
        rotor_E_cross_product = np.cross(rotor_E_relative_positions, axis_E_vector)

        axis_F_vector = self.positions[axis_ring_F[1]] - self.positions[axis_ring_F[0]]
        axis_F_center = (self.positions[axis_ring_F[1]] + self.positions[axis_ring_F[0]]) / 2.0
        rotor_F_relative_positions = self.positions[ring_F] - axis_F_center
        rotor_F_cross_product = np.cross(rotor_F_relative_positions, axis_F_vector)

        
        forces_torque_1 = rotor_1_cross_product * self.forces_alpha[0]
        forces_torque_2 = rotor_2_cross_product * self.forces_alpha[1]
        forces_torque_3 = rotor_3_cross_product * self.forces_alpha[2]
        forces_torque_4 = rotor_4_cross_product * self.forces_alpha[3]
        forces_torque_5 = rotor_5_cross_product * self.forces_alpha[4]
        forces_torque_6 = rotor_6_cross_product * self.forces_alpha[5]

        forces_torque_A = rotor_A_cross_product * self.forces_alpha[6]
        forces_torque_B = rotor_B_cross_product * self.forces_alpha[7]
        forces_torque_C = rotor_C_cross_product * self.forces_alpha[8]
        forces_torque_D = rotor_D_cross_product * self.forces_alpha[9]
        forces_torque_E = rotor_E_cross_product * self.forces_alpha[10]
        forces_torque_F = rotor_F_cross_product * self.forces_alpha[11]


        forces[ring_1 ] += forces_torque_1
        forces[ring_2 ] += forces_torque_2
        forces[ring_3 ] += forces_torque_3
        forces[ring_4 ] += forces_torque_4
        forces[ring_5 ] += forces_torque_5
        forces[ring_6 ] += forces_torque_6

        forces[ring_A ] += forces_torque_A
        forces[ring_B ] += forces_torque_B
        forces[ring_C ] += forces_torque_C
        forces[ring_D ] += forces_torque_D
        forces[ring_E ] += forces_torque_E
        forces[ring_F ] += forces_torque_F

        # ic(forces[ring_1])
        # ic(rotor_1_cross_product)
        # ic(rotor_2_cross_product)
        # ic(rotor_3_cross_product)
        # ic(rotor_4_cross_product)
        # ic(rotor_5_cross_product)
        # ic(rotor_6_cross_product)
        # ic(rotor_A_cross_product)
        # ic(rotor_B_cross_product)
        # ic(rotor_C_cross_product)
        # ic(rotor_D_cross_product)
        # ic(rotor_E_cross_product)
        # ic(rotor_F_cross_product)

        # ic(self.positions[ring_1])
        # ic(self.positions[ring_2])
        # ic(self.positions[ring_3])
        # ic(self.positions[ring_4])
        # ic(self.positions[ring_5])
        # ic(self.positions[ring_6])
        # ic(self.positions[ring_A])
        # ic(self.positions[ring_B])
        # ic(self.positions[ring_C])
        # ic(self.positions[ring_D])
        # ic(self.positions[ring_E])
        # ic(self.positions[ring_F])

        from ase.visualize import view
        # view(self[ring_3_full])
        # input("Press Enter to continue...")
        # ic(self.get_chemical_symbols())

        # ic(self.get_cell())

        from ase import io
        io.write("/home/dlbox2/repos/rotor-gp/code/pimd/355K_dftb_1/sample_crystal.xyz", self, format="extxyz")
        input("Press Enter to continue...")
        crys = io.read("/home/dlbox2/repos/rotor-gp/code/pimd/355K_dftb_1/sample_crystal.xyz", format="extxyz")
        view(crys)

        return forces

####




# import sys
# sys.path.append("../../fande")
# from prepare_model import prepare_fande_ase_calc
# from fande.ase import FandeAtomsWrapper 

from xtb.ase.calculator import XTB
# from ase.calculators.gulp import GULP, Conditions
# os.environ["ASE_GULP_COMMAND"]="/home/qklmn/data/gulp/gulp-6.1.2/Src/gulp < PREFIX.gin > PREFIX.got"
# /home/qklmn/data/gulp/gulp-6.1.2/Libraries
# os.environ["GULP_LIB"] = "/home/qklmn/data/gulp/gulp-6.1.2/Libraries"

os.environ['OMP_NUM_THREADS'] = "1,1"
# os.environ["ASE_DFTB_COMMAND"] = "ulimit -s unlimited; /usr/local/dftbplus-21.2/bin/dftb+ > PREFIX.out"
# os.environ["ASE_DFTB_COMMAND"] = "ulimit -s unlimited; mpirun -np 1 dftb+ > PREFIX.out"
os.environ["ASE_DFTB_COMMAND"] = "ulimit -s unlimited; dftb+ > PREFIX.out"
#with mpi
# os.environ["ASE_DFTB_COMMAND"] = "source ~/.bashrc; mpirun -np 20 -ppn 1 dftb+-mpi > PREFIX.out"
# os.environ["ASE_DFTB_COMMAND"] = "dftb+ > PREFIX.out"
# os.environ["DFTB_PREFIX"] =  #"/home/qklmn/data/dftb/pbc-0-3"
os.environ["DFTB_PREFIX"] = "/home/dlbox2/ダウンロード/pbc-0-3"

def make_client(i, gpu_id_list):
    pimd_dirname = os.environ.get('PIMD_DIR')
    if pimd_dirname is None:
        pimd_dirname = 'caclulation_temp' ############### specify correctly!!!!!
    temp_dir = "pimd/" + pimd_dirname + "/calc_" + str(PIMD_PORT) + "_" + str(i)
    os.makedirs(temp_dir, exist_ok=True)
    os.chdir(temp_dir)
    # for file in os.scandir(temp_dir):
    #     os.remove(file.path)


    pimd_port = PIMD_PORT
        # pimd_port = pimd_port_list[i]

    atoms_copy = atoms.copy()

    # atoms_copy = FandeAtomsWrapper(atoms_copy)
    # atoms_copy.request_variance = True
    # fande_calc = prepare_fande_ase_calc(hparams, soap_params, gpu_id = gpu_id_list[i])
    # calc = fande_calc
   
    # https://dftb.org/parameters/download/3ob/3ob-3-1-cc
    # atoms_copy = FandeAtomsWrapper(atoms_copy)
    # atoms_copy.request_variance = False
    atoms_copy = RotationAtomsWrapper(atoms_copy)

    # atoms_copy.set_pbc(False)
    # calc_xtb = XTB(method='GFN-FF')
    # atoms_copy.set_calculator(calc_xtb)

    # atoms_copy.set_pbc(False)
    # calc_gulp = GULP(keywords='gfnff gwolf conv gradient', options=[''], library=False)
    # atoms_copy.set_calculator(calc_gulp)

    atoms_copy.set_pbc(True)
    calc = Dftb(atoms=atoms_copy,
            label='crystal',
            # Hamiltonian_ = "xTB",
            # # Hamiltonian_Method = "GFN1-xTB",
            Hamiltonian_MaxAngularMomentum_='',
            Hamiltonian_MaxAngularMomentum_H='s',
            Hamiltonian_MaxAngularMomentum_O='p',
            Hamiltonian_MaxAngularMomentum_N='p',
            Hamiltonian_MaxAngularMomentum_C='p',
            Hamiltonian_MaxAngularMomentum_Si='d',
            kpts=(1,1,1),
            # Hamiltonian_SCC='Yes',
            # Verbosity=0,
            # Hamiltonian_OrbitalResolvedSCC = 'Yes',
            # Hamiltonian_SCCTolerance=1e-15,
            # kpts=None,
            # Driver_='ConjugateGradient',
            # Driver_MaxForceComponent=1e-3,
            # Driver_MaxSteps=200,
            # Driver_LatticeOpt = 'Yes',
            #     Driver_AppendGeometries = 'Yes',
            #     Driver_='',
            #     Driver_Socket_='',
            #     Driver_Socket_File='Hello'
            )

    atoms_copy.set_calculator(calc)

    print( atoms_copy.get_forces() )

    print("Calculator is set up!")
    # print( atoms_copy.get_forces() )
    # Create Client
    # inet
    print(f"Launching fande client at port {pimd_port}")
    port = pimd_port
    host = "localhost"
    client = SocketClient(host=host, port=port)
    client.run(atoms_copy)

    return 0



from joblib import Parallel, delayed

K = PIMD_NUM_CALCS #len(pimd_port_list)
gpu_id_list = []
# gpu_id_list = [0, 1, 2, 3, 4, 5, 6, 7] * 2
# K=41

print("Starting clients with joblib...")
status = Parallel(n_jobs=K, prefer="processes")(delayed(make_client)(i, gpu_id_list) for i in range(0, K)) 


# import datetime

# now_dir = str(datetime.datetime.now())
# now_dir = now_dir.replace(" ", "_")
# now_dir = "dump/run_calc_history_" +  now_dir

# for i in range(0, K):
#     new_dir = now_dir + "/ase_calc_history_" + str(i)
#     # print(new_dir)
#     os.makedirs(new_dir, exist_ok=True)
#     os.system("cp temp/temp_calc_dir_" + str(i) + "/ase_calc_history/* " + new_dir + "/")
# os.system(f"for file in PREFIX*; do mv $file  {now_dir}/; done" )
# os.system(f"for file in *log*; do cp $file  {now_dir}/; done" )
# os.system(f"for file in *HILLS*; do mv $file  {now_dir}/; done" )
# os.system(f"for file in *COLVAR*; do mv $file  {now_dir}/; done" )
# os.system(f"for file in *RESTART*; do mv $file  {now_dir}/; done" )
# os.system(f"for file in *input*.xml; do cp $file  {now_dir}/; done" )
# os.system(f"cp -r plumed  {now_dir}/plumed; done" )


print(f"Finished {K} clients with statuses: ", status)

# with SocketIOCalculator(calc_base, log=sys.stdout, unixsocket='Hello') as calc:
#         atoms.set_calculator(calc)
#         client.run(atoms)

# # ################# Create ASE SERVER ############################
# https://github.com/i-pi/i-pi/blob/master/examples/ASEClient/aims_double_server/run-ase.py

# with SocketIOCalculator(calc_base, log="socketio.log", port=port) as io_calc:
#     atoms.set_calculator(io_calc)
#     client.run(atoms)