import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



class AtomGroups:
    def __init__(self, universe, atom_list, resname):
        self.universe = universe
        self.atom_list = atom_list
        self.resname = resname
        self.group1_indices = []
        self.group2_indices = []

    def _get_valid_indices(self):
        all_molecules = self.universe.select_atoms(f"resname {self.resname}", updating=True)
        for molecule in all_molecules.residues:
            atoms_in_molecule = molecule.atoms
            if all(atom.index in all_molecules.indices for atom in atoms_in_molecule):
                self.group1_indices.extend(molecule.atoms.select_atoms("name " + " ".join(self.atom_list[0][0])).indices)
                for atom_group in self.atom_list[0][1:]:
                    self.group2_indices.extend(molecule.atoms.select_atoms("name " + " ".join(atom_group)).indices)

    def get_atom_groups(self):
        self._get_valid_indices()
        group1 = self.universe.atoms[self.group1_indices]
        group2 = self.universe.atoms[self.group2_indices]
        return group1, group2


class OutputHandler:
    @staticmethod
    def save_to_file(filename, data):
        df = pd.DataFrame(data)
        df.to_csv(filename, index=False)
        np.savetxt(filename, data)

class OrderParameters:
    def __init__(self, universe, atom_list, resname, selection):
        self.universe = universe
        self.atom_list = atom_list
        self.resname = resname
        self.selection = selection
        self.atom_groups = AtomGroups(self.universe, self.atom_list, self.resname)

    def compute_OP(self):


        group1, group2 = self.atom_groups.get_atom_groups()

        C_numbers = [int(atom.name[1:]) for atom in group1] # extract carbon numbers from group1

        natoms = len(self.atom_list[0])  # How many center atoms
        nmols = int(len(group1.positions)/natoms)  # How many molecules  
        repeat = [len(atoms) - 1 for atoms in self.atom_list] # How many hydrogen atoms per center carbon atom
        repeats = repeat * nmols  # Calculate repeats inside loop

        output = []
        for ts in self.universe.trajectory:
            p1 = np.repeat(group1.positions, repeats, axis=0)
            p2 = group2.positions
            dp = p2 - p1
            norm = np.sqrt(np.sum(np.power(dp, 2), axis=-1))
            cos_theta = dp[...,2]/norm
            S = -0.5 * (3 * np.square(cos_theta) - 1)

            new_S = self._repeat(S, repeats)
            new_S.shape = (nmols, natoms)
            results = np.average(new_S, axis=0)
            output.append(results)

        avg = np.average(output, axis=0)
        result = np.transpose([C_numbers, avg])
        return result

    def _repeat(self, x, reps):
        ''' 
        x = [1, 3, 5, 7, 9]
        reps = [2, 2, 1]
        out = [2, 6, 9]
        '''
        assert len(x) == int(np.sum(reps)), 'repeats wrong'

        i = 0
        out = []
        for rep in reps:
            tmp = []
            for r in range(rep):
                tmp.append(x[i])
                i += 1
            out.append(np.average(tmp))
    
        b = np.array(out)
        # print(b,'b')
        return b

    def save_results(self, filename, data):
        OutputHandler.save_to_file(filename, data)




import opc
import MDAnalysis as mda

u = mda.Universe('../trajs/run.gro', '../trajs/test_dcd.xtc')
halfz = u.dimensions[2] / 2

selection = ('resname POPC and not around 30 protein and prop z > %f' % halfz)
selection_DOPE = ('resname DOPE and not around 30 protein and prop z > %f' % halfz)

op_POPC1 = OrderParameters(u, opc.POPC1, 'POPC', selection)
POPC1 = op_POPC1.compute_OP()
op_POPC1.save_results('POPC1_top_long.dat', POPC1)

op_POPC2 = OrderParameters(u, opc.POPC2, 'POPC', selection)
POPC2 = op_POPC2.compute_OP()
op_POPC2.save_results('POPC2_top_long.dat', POPC2)

op_DOPE1 = OrderParameters(u, opc.DOPE1, 'DOPE', selection_DOPE)
DOPE1 = op_DOPE1.compute_OP()
op_DOPE1.save_results('DOPE1_top_long.dat', DOPE1)
