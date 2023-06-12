# To run the server: i-pi input.xml > log &
# To run the driver: python ./run_driver.py

# i-pi input.xml > log || python ./run_driver.py

# To clean up: 
# rm -f *log* *PREFIX*  *RESTART* *COLVAR* plumed/*COLVAR* plumed/*HILLS* *HILLS* *colvar* 

# Tutorial:
# https://gitlab.com/Sucerquia/ase-plumed_tutorial


import os
import sys

from ase.calculators.socketio import SocketClient
from ase.io import read
from ase import units

from ase.calculators.lj import LennardJones
from xtb.ase.calculator import XTB


# Define atoms object
atoms = read("init.xyz", index='0', format='extxyz')


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
calc_base = XTB(method="GFN2-xTB")

# calc = Plumed(calc=calc_base,
#                     input=setup,
#                     timestep=timestep,
#                     atoms=atoms,
#                     kT=0.1)


atoms.set_calculator(calc_base)
print("Calculator is set up!")


# Create Client
# inet
port = 10200
host = "localhost"
client = SocketClient(host=host, port=port)

client.run(atoms)