
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class OrderParameters:
    def __init__(self, u, atomlists, resname, selection):
        self.u = u
        self.atomlists = atomlists
        self.resname = resname
        self.selection = selection
        self.C_numbers, self.Cs, self.Hs_f, self.repeat = self.process_atom_lists()

    def process_atom_lists(self):
        C_numbers = []
        Cs = []
        Hs = []
        repeat = []
        for atoms in self.atomlists:
            C_number = atoms[0][2:]
            C_numbers.append(int(C_number))
            Cs.append(atoms[0])
            Hs.append(atoms[1:])
            repeat.append(len(atoms) - 1)  # How many hydrogen atoms per center carbon atom
        Hs_f = [item for sublist in Hs for item in sublist]
        assert int(np.sum(repeat)) == len(Hs_f), "Wrong in repeats"
        return C_numbers, Cs, Hs_f, repeat

    def compute_OP(self):
        all_molecules = self.u.select_atoms(self.selection, updating=True)
        output = []

        for ts in self.u.trajectory:
            valid_indices_group1 = []
            valid_indices_group2 = []
            for molecule in all_molecules.residues:
                atoms_in_molecule = molecule.atoms
                if all(atom.index in all_molecules.indices for atom in atoms_in_molecule):
                    valid_indices_group1.extend(molecule.atoms.select_atoms("name " + " ".join(self.Cs)).indices)
                    valid_indices_group2.extend(molecule.atoms.select_atoms("name " + " ".join(self.Hs_f)).indices)
            group1 = self.u.atoms[valid_indices_group1]
            print(len(group1))
            group2 = self.u.atoms[valid_indices_group2]

            natoms = len(self.Cs)  # How many center atoms
            nmols = int(len(group1.positions) / natoms)  # How many molecules
            repeats = self.repeat * nmols  # Calculate repeats inside loop

            p1 = np.repeat(group1.positions, repeats, axis=0)
            p2 = group2.positions
            dp = p2 - p1
            norm = np.sqrt(np.sum(np.power(dp, 2), axis=-1))
            cos_theta = dp[..., 2] / norm
            S = -0.5 * (3 * np.square(cos_theta) - 1)

            new_S = self._average_over_hydrogens(S, repeats)
            new_S.shape = (nmols, natoms)
            results = np.average(new_S, axis=0)
            output.append(results)

        avg = np.average(output, axis=0)
        return np.transpose([self.C_numbers, avg])

    def _average_over_hydrogens(self, x, reps):
        """
        Averages over the number of hydrogen atoms and returns a numpy array.

        Parameters
        ----------
        x : np.ndarray
            An array containing some property of the hydrogen atoms.
        reps : list
            A list containing the number of times each hydrogen atom appears.

        Returns
        -------
        np.ndarray
            A numpy array containing the average of the property over the hydrogen atoms.
        """
        assert len(x) == int(np.sum(reps)), 'Repeats wrong'
        i = 0
        out = []
        for rep in reps:
            tmp = []
            for r in range(rep):
                tmp.append(x[i])
                i += 1
            out.append(np.average(tmp))
        return np.array(out)


import opc
import MDAnalysis as mda

u = mda.Universe('../trajs/run.gro', '../trajs/test_dcd.xtc')
halfz = u.dimensions[2] / 2


# Change selection to 'around' or 'not around' protein depending if you want far or close to protein
selection = ('resname POPC and around 30 protein and prop z < %f' % halfz)
# selection_DOPE = ('resname DOPE and not around 10 protein and prop z < %f' % halfz)

OP_POPC1 = OrderParameters(u, opc.POPC1, 'POPC', selection)
POPC1 = OP_POPC1.compute_OP()

# OP_POPC2 = OrderParameters(u, opc.POPC2, 'POPC', selection)
# POPC2 = OP_POPC2.compute_OP()

# OP_DOPE1 = OrderParameters(u, opc.DOPE1, 'DOPE', selection_DOPE)
# DOPE1 = OP_DOPE1.compute_OP()

np.savetxt('POPC1_bottom_3nm_close.dat', POPC1)
# np.savetxt('POPC2_top_long.dat', POPC2)
# np.savetxt('DOPE1_bottom_1nm.dat', DOPE1)

