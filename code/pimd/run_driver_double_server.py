# To run the server: i-pi input.xml > log &
# To run the driver: python ./run_driver.py

# i-pi input.xml > log || python ./run_driver.py

# To clean up: 
# rm -f *log* *PREFIX*  *RESTART* *COLVAR* plumed/*COLVAR* plumed/*HILLS* *HILLS* *colvar* 

# Tutorial:
# https://gitlab.com/Sucerquia/ase-plumed_tutorial
# https://dftbplus-recipes.readthedocs.io/en/latest/interfaces/ipi/ipi.html


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
# print(atoms.get_chemical_symbols())




# # from plumed import Plumed
# from runner.plumed import Plumed
# timestep = 0.005
# ps = 1000 * units.fs
# setup = [f"UNITS LENGTH=A TIME={1/(1000 * units.fs)} ENERGY={units.mol/units.kJ}",
#          "d: DISTANCE ATOMS=4,5",
#          "mtd:   METAD ARG=d PACE=6 SIGMA=0.1 HEIGHT=4 FILE=plumed/HILLS BIASFACTOR=10 TEMP=300",
#          "PRINT ARG=d STRIDE=10 FILE=plumed/COLVAR"]
# # print(1/(1000 * units.fs))


# Set up the calculator #################
# calc_base = XTB(method="GFN2-xTB")
# calc = Plumed(calc=calc_base,
#                     input=setup,
#                     timestep=timestep,
#                     atoms=atoms,
#                     kT=0.1)



import os
os.environ['OMP_NUM_THREADS'] = "6,1"
os.environ["ASE_DFTB_COMMAND"] = "ulimit -s unlimited; /home/dlbox2/anaconda3/envs/fande/bin/dftb+ > PREFIX.out"
os.environ["DFTB_PREFIX"] = "./pbc-0-3"
calc_base = Dftb(atoms=atoms,
            label='crystal',
            # Hamiltonian_ = "xTB",
            # Hamiltonian_Method = "GFN1-xTB",
            Hamiltonian_MaxAngularMomentum_='',
            Hamiltonian_MaxAngularMomentum_O='p',
            Hamiltonian_MaxAngularMomentum_H='s',
            Hamiltonian_MaxAngularMomentum_N='s',
            Hamiltonian_MaxAngularMomentum_C='s',
            Hamiltonian_MaxAngularMomentum_Si='s',
            kpts=(1,1,1),
            # Hamiltonian_SCC='Yes',
            # Verbosity=0,
            # Hamiltonian_OrbitalResolvedSCC = 'Yes',
            # Hamiltonian_SCCTolerance=1e-15,
            # kpts=None
            # Driver_='ConjugateGradient',
            # Driver_MaxForceComponent=1e-3,
            # Driver_MaxSteps=200,
            # Driver_LatticeOpt = 'Yes',
            # Driver_AppendGeometries = 'Yes'
            Driver_='',
            Driver_Socket_='',
            Driver_Socket_File='Hello'
            )

# atoms.set_calculator(calc_base)
print("Calculator is set up!")


# Create Client
# inet
port = 10200
host = "localhost"
client = SocketClient(host=host, port=port)

# client.run(atoms)


with SocketIOCalculator(calc_base, log=sys.stdout, unixsocket='Hello') as calc:
        atoms.set_calculator(calc)
        client.run(atoms)

print("Finished running!")

# # ################# Create ASE SERVER ############################
# https://github.com/i-pi/i-pi/blob/master/examples/ASEClient/aims_double_server/run-ase.py
# https://databases.fysik.dtu.dk/ase/ase/calculators/socketio/socketio.html

# with SocketIOCalculator(calc_base, log="socketio.log", port=port) as io_calc:
#     atoms.set_calculator(io_calc)
#     client.run(atoms)