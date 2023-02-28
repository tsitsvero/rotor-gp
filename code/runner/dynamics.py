from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase import units
from ase import io
import os


class MDRunner:
    def __init__(self, atoms, traj_filename, log_filename):
        self.atoms = atoms
        self.traj_filename = traj_filename
        self.log_filename = log_filename

    def run(self, dt=0.01, num_steps=100):
        # atoms.calc = EMT()

        # Set the momenta corresponding to T=300K
        MaxwellBoltzmannDistribution(self.atoms, temperature_K=300)

        # momenta = self.atoms.get_momenta()
        # momenta[11] = [0.1,0.1,0.1]
        # self.atoms.set_momenta(momenta)

        # We want to run MD with constant energy using the VelocityVerlet algorithm.
        os.makedirs("../dump/ase", exist_ok=True)
        dyn = VelocityVerlet(
            self.atoms,
            dt = dt,
            trajectory=self.traj_filename,
            logfile=self.log_filename,
        )  # 5 fs time step.

        # def printenergy(a):
        #     """Function to print the potential, kinetic and total energy"""
        #     epot = a.get_potential_energy() / len(a)
        #     ekin = a.get_kinetic_energy() / len(a)
        #     print(
        #         "Energy per atom: Epot = %.3feV  Ekin = %.3feV (T=%3.0fK)  "
        #         "Etot = %.3feV" % (epot, ekin, ekin / (1.5 * units.kB), epot + ekin)
        #     )

        dyn.run(num_steps)


        # # Now run the dynamics
        # printenergy(atoms)
        # for i in range(5):
        #     dyn.run(10)
        #     # printenergy(atoms)

        # traj = io.read(os.path.dirname(self.traj_filename) + "md.traj", index=":")
        # io.write(self.traj_filename, traj)