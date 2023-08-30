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
os.environ['OMP_NUM_THREADS'] = "5,1"
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
'interaction_cutoff': 5.0, #5
'gaussian_sigma_constant': 0.3,
'max_radial': 5, #5
'max_angular': 5,#5
'cutoff_smooth_width': 0.5,
# 'average': "off",
# 'crossover': True,
# 'dtype': "float64",
# 'n_jobs': 10,
# 'sparse': False,
# 'positions': [7, 11, 15] # ignored
}

import sys
sys.path.append("../../fande")

def make_client(i, gpu_id_list, prefix='dftb_good_1_'):
    temp_dir = "results/temp/" + prefix + str(i)
    os.makedirs(temp_dir, exist_ok=True)
    os.chdir(temp_dir)
    # for file in os.scandir(temp_dir):
    #     os.remove(file.path)


    atoms_copy = atoms.copy()


    # atoms_copy = FandeAtomsWrapper(atoms_copy, request_uncertainties=True)
    # atoms_copy.request_uncertainties = True
    # fande_calc = prepare_fande_ase_calc(hparams, soap_params, gpu_id = gpu_id_list[i])
    # calc = fande_calc
   

    # https://dftb.org/parameters/download/3ob/3ob-3-1-cc
    atoms_copy = FandeAtomsWrapper(atoms_copy)
    atoms_copy.request_variance = False
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
            kpts=(2,1,1),
            Hamiltonian_SCC='Yes',
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
    port = 10201
    host = "localhost"
    client = SocketClient(host=host, port=port)
    client.run(atoms_copy)

    return 0



from joblib import Parallel, delayed

K = 16
gpu_id_list = [0, 1, 2, 3, 4, 5, 6, 7] * 2
K=1

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