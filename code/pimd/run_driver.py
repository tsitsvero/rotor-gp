# To run the server: i-pi input.xml > log &
# To run the driver: python ./run_driver.py

# i-pi input.xml > log || python ./run_driver.py

# To clean up: 
# rm -rf *log* *PREFIX*  *RESTART* *COLVAR* plumed/*COLVAR* plumed/*HILLS* *HILLS* *colvar* temp_calc_dir_* EXIT

# Tutorial:
# https://gitlab.com/Sucerquia/ase-plumed_tutorial
# https://dftbplus-recipes.readthedocs.io/en/latest/interfaces/ipi/ipi.html



from ase import Atoms


class AtomsWrapped(Atoms):
    
    def __init__(self, *args, **kwargs):
        super(AtomsWrapped, self).__init__(*args, **kwargs
                                                 )
       
        self.calc_history_counter = 0


    def get_forces(self):       
        forces = super(AtomsWrapped, self).get_forces()
        # energy = super(AtomsWrapped, self).get_potential_energy()

        from ase.io import write
        import os

        os.makedirs("ase_calc_history" , exist_ok=True)
        write( "ase_calc_history/" + str(self.calc_history_counter) + ".xyz", self, format="extxyz")
        # self.calc_history.append(self.copy())       
        self.calc_history_counter += 1

        return forces
    
    





import os
import sys

from ase.calculators.socketio import SocketClient, SocketIOCalculator
from ase.io import read
from ase import units

# from ase.calculators.lj import LennardJones
# from xtb.ase.calculator import XTB
from ase.calculators.dftb import Dftb


# Define atoms object
# atoms = read("init.xyz", index='0', format='extxyz')
# atoms = read(' ../../../structures/structures/new_systems/ktu_002.xyz')
atoms = read('/home/dlbox2/repos/structures/structures/new_systems/ktu_002.cif')
atoms = AtomsWrapped(atoms)



# Set up the calculator #################
# calc_base = XTB(method="GFN2-xTB")
# calc = Plumed(calc=calc_base,
#                     input=setup,
#                     timestep=timestep,
#                     atoms=atoms,
#                     kT=0.1)


import os
os.environ['OMP_NUM_THREADS'] = "6,1"
os.environ["ASE_DFTB_COMMAND"] = "ulimit -s unlimited; dftb+ > PREFIX.out"
os.environ["DFTB_PREFIX"] = "/home/dlbox2/ダウンロード/pbc-0-3"



def make_client(i):
    os.makedirs("temp/temp_calc_dir_" + str(i), exist_ok=True)
    os.chdir("temp/temp_calc_dir_" + str(i))

    atoms_copy = atoms.copy()

    calc = Dftb(atoms=atoms_copy,
                label='crystal',
                # Hamiltonian_ = "xTB",
                # Hamiltonian_Method = "GFN1-xTB",
            #     Hamiltonian_MaxAngularMomentum_='',
            #     Hamiltonian_MaxAngularMomentum_O='p',
            #     Hamiltonian_MaxAngularMomentum_H='s',
            #     Hamiltonian_MaxAngularMomentum_N='s',
            #     Hamiltonian_MaxAngularMomentum_C='s',
            #     Hamiltonian_MaxAngularMomentum_Si='s',
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

    print("Calculator is set up!")
    # Create Client
    # inet
    port = 10200
    host = "localhost"
    client = SocketClient(host=host, port=port)
    client.run(atoms_copy)

    return 0



from joblib import Parallel, delayed

K = 32

status = Parallel(n_jobs=K, prefer="processes")(delayed(make_client)(i) for i in range(0, K)) 


import datetime

now = datetime.datetime.now()


for i in range(0, K):
    new_dir = "dump/run_calc_history_" + str(now) + "/ase_calc_history_" + str(i)
    new_dir = new_dir.replace(" ", "_")
    # print(new_dir)
    os.makedirs(new_dir, exist_ok=True)
    os.system("cp temp/temp_calc_dir_" + str(i) + "/ase_calc_history/* " + new_dir + "/")


print(f"Finished {K} clients with statuses: ", status)

# with SocketIOCalculator(calc_base, log=sys.stdout, unixsocket='Hello') as calc:
#         atoms.set_calculator(calc)
#         client.run(atoms)


# # ################# Create ASE SERVER ############################
# https://github.com/i-pi/i-pi/blob/master/examples/ASEClient/aims_double_server/run-ase.py

# with SocketIOCalculator(calc_base, log="socketio.log", port=port) as io_calc:
#     atoms.set_calculator(io_calc)
#     client.run(atoms)