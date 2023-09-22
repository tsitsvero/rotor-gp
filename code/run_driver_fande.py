# To run the server: i-pi input.xml > log &
# To run the driver: python ./run_driver.py

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
    



import os
import sys

from ase.calculators.socketio import SocketClient, SocketIOCalculator
from ase.io import read,write
from ase import units

# from ase.calculators.lj import LennardJones
# from xtb.ase.calculator import XTB
from ase.calculators.dftb import Dftb


import sys
sys.path.append("../../fande")
from prepare_model import prepare_fande_ase_calc
from fande.ase import FandeAtomsWrapper 


# Define atoms object
# atoms = read("init.xyz", index='0', format='extxyz')
# atoms = read(' ../../../structures/structures/new_systems/ktu_002.xyz')

# atoms = read('/home/qklmn/repos/structures/structures/new_systems/ktu_002.cif')
atoms = read('/home/qklmn/data/starting_configuration/1.cif') # atoms specified here should be the same as in i-pi input file (otherwise atomic order differ, structure blows up!)



# Set up the calculator #################
# calc_base = XTB(method="GFN2-xTB")
# calc = Plumed(calc=calc_base,
#                     input=setup,
#                     timestep=timestep,
#                     atoms=atoms,
#                     kT=0.1)

import os
os.environ['OMP_NUM_THREADS'] = "3,1"
os.environ["ASE_DFTB_COMMAND"] = "ulimit -s unlimited; /usr/local/dftbplus-21.2/bin/dftb+ > PREFIX.out"
# os.environ["ASE_DFTB_COMMAND"] = "dftb+ > PREFIX.out"
os.environ["DFTB_PREFIX"] = "/home/qklmn/data/dftb/pbc-0-3"

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

from ase import Atoms
# from ase.io import write



# import os
import numpy as np

axis_ring = [111,11]
ring = [25, 27, 29, 31, 33, 35]
class RotationAtomsWrapper(Atoms):   
    def __init__(self, *args, **kwargs):
        super(RotationAtomsWrapper, self).__init__(*args, **kwargs)      
        self.calc_history_counter = 0
        self.alpha = 0.5

    def get_forces(self, md=False):       
        forces = super(RotationAtomsWrapper, self).get_forces(md=md)
        # energy = super(AtomsWrapped, self).get_potential_energy()
        # os.makedirs("ase_calc_history" , exist_ok=True)
        # write( "ase_calc_history/" + str(self.calc_history_counter) + ".xyz", self, format="extxyz")
        # self.calc_history.append(self.copy())       
        # self.calc_history_counter += 1

        axis_1_vector = self.positions[axis_ring[1]] - self.positions[axis_ring[0]]
        axis_1_center = (self.positions[axis_ring[1]] + self.positions[axis_ring[0]]) / 2.0
        rotor_1_relative_positions = self.positions[ring] - axis_1_center
        rotor_1_cross_product = np.cross(rotor_1_relative_positions, axis_1_vector)

        # axis_2_vector = self.positions[axis_bulk[1]] - self.positions[axis_bulk[0]]
        # axis_2_center = (self.positions[axis_bulk[1]] + self.positions[axis_bulk[0]]) / 2.0
        # rotor_2_relative_positions = self.positions[bulk_1 + bulk_2 + bulk_3] - axis_2_center
        # rotor_2_cross_product = np.cross(rotor_2_relative_positions, axis_2_vector)


        forces_torque_1 = rotor_1_cross_product * self.alpha
        # forces_torque_2 = rotor_2_cross_product * self.alpha

        forces[ring] += forces_torque_1
        # forces[bulk_1 + bulk_2 + bulk_3] += -forces_torque_2
        # forces[frozen_atoms] = np.zeros((len(frozen_atoms), 3))

        return forces

####



import sys
sys.path.append("../../fande")

def make_client(i, gpu_id_list):
    pimd_dirname = os.environ.get('PIMD_DIR')
    if pimd_dirname is None:
        pimd_dirname = 'output_ml_16' ############### specify correctly!!!!!
    temp_dir = "pimd/" + pimd_dirname + "/calc_" + str(i)
    os.makedirs(temp_dir, exist_ok=True)
    os.chdir(temp_dir)
    # for file in os.scandir(temp_dir):
    #     os.remove(file.path)


    if os.environ.get('PIMD_PORT') is not None:
        pimd_port = int(os.environ.get('PIMD_PORT'))
    else:
        pimd_port = 10200


    atoms_copy = atoms.copy()


    # atoms_copy = FandeAtomsWrapper(atoms_copy)
    # atoms_copy.request_variance = True
    # fande_calc = prepare_fande_ase_calc(hparams, soap_params, gpu_id = gpu_id_list[i])
    # calc = fande_calc
   





    # https://dftb.org/parameters/download/3ob/3ob-3-1-cc
    atoms_copy = FandeAtomsWrapper(atoms_copy)
    atoms_copy.request_variance = False
    atoms_copy = RotationAtomsWrapper(atoms_copy)
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

    print( atoms_copy.get_potential_energy() )

    print("Calculator is set up!")
    # print( atoms_copy.get_forces() )
    # Create Client
    # inet
    print(f"Launchine fande client at port {pimd_port}")
    port = pimd_port
    host = "localhost"
    client = SocketClient(host=host, port=port)
    client.run(atoms_copy)

    return 0



from joblib import Parallel, delayed

K = 16
gpu_id_list = [0, 1, 2, 3, 4, 5, 6, 7] * 2
# K=1

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