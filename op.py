import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class OrderParameters:
    def __init__(self):
        pass
    
    def compute_OP(self, u, atomlists = [], resname = False, selection=''):
        assert resname, 'provide a resname'
        
        C_numbers = []
        Cs = []
        Hs = []
        repeat = []
        for atoms in atomlists:
            C_number = atoms[0][2:]
            C_numbers.append(int(C_number))
            
            Cs.append(atoms[0])
            Hs.append(atoms[1:])
            ## How many hydrogen atoms per center carbon atom
            repeat.append(len(atoms)-1) 


        Hs_f = [item for sublist in Hs for item in sublist]
        assert int(np.sum(repeat)) == len(Hs_f), "wrong in repeats"
        ## Cs =   [C32, C33, ..., C318]
        ## Hs =   [[H2X, H2Y], [H3X, H3Y], ... [H18X, H18Y, H18Z]]
        ## Hs_f = [H2X, H2Y, H3X, ... , H18Z]
        ## repeat = [2, 2, 2, ..., 3]

        # Select all atoms that match the provided resname
        # all_molecules = u.select_atoms(f"resname {resname}")
        all_molecules = u.select_atoms(selection, updating=True)

        # Create an empty AtomGroup using a select_atoms() call with an impossible condition
        empty_group = u.select_atoms('')

        # Use this empty_group as your initial AtomGroup for storing the atoms that pass your condition
        valid_molecules_group1 = empty_group
        valid_molecules_group2 = empty_group

        # Iterate over each residue in all_molecules
        for molecule in all_molecules.residues:
            # Get all the atoms in the current molecule
            atoms_in_molecule = molecule.atoms

            # The select_atoms call with the condition is put in a loop over the atoms_in_molecule
            # This checks that every atom in atoms_in_molecule satisfies the condition
            # If every atom in the molecule passes the condition, the if condition will be True
            if all(u.select_atoms(f"(resname {resname}) and bynum {atom.index + 1}") for atom in atoms_in_molecule):
                # Only include the molecule if all its atoms satisfy the condition
                # Select the atoms in the molecule that match the names in Cs
                valid_molecules_group1 += molecule.atoms.select_atoms("name " + " ".join(Cs))

                # Select the atoms in the molecule that match the names in Hs_f
                valid_molecules_group2 += molecule.atoms.select_atoms("name " + " ".join(Hs_f))

        # Now, 'valid_molecules_group1' and 'valid_molecules_group2' only contain atoms from complete molecules that satisfy your condition
        # Convert the list of valid molecules to AtomGroups
        group1 = valid_molecules_group1
        group2 = valid_molecules_group2



        natoms       = len(Cs) ## How many center atoms
        nmols        = int(len(group1.positions)/natoms) ## How many molecules
        repeats = repeat * nmols
        # repeats      = repeat * nmols ## [2, 2, 2, ..., 3, | 2, 2, 2, ..., 3, | ...]
        splits       = np.cumsum(repeats)
        
        print('# of mols: %d' %nmols)
        print('# of Carbons per molecule: %d' %natoms)

        output = []
        for ts in u.trajectory:
            print(len(group1))
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
        
        # print(np.array(output))
        avg = np.average(output, axis=0)
        np.transpose([C_numbers, avg])
        a = np.transpose([C_numbers, avg])
        return a
        # print(a)


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
        

import opc
import MDAnalysis as mda
u = mda.Universe('../trajs/run.gro', '../trajs/test_dcd.xtc')
halfz = u.dimensions[2] / 2

selection = ('resname POPC and not around 30 protein and prop z > %f' % halfz)
# selection = ('resname POPC and prop z > %f' % halfz)
# selection = ('resname POPC')

POPC1 = OrderParameters().compute_OP(u, opc.POPC1, resname ='POPC', selection = selection)
# POPC2 = OrderParameters().compute_OP(u, opc.POPC2, resname = 'POPC')
#DOPE1 = OrderParameters().compute_OP(u, opc.DOPE1, resname = 'DOPE')
#DOPE2 = OrderParameters().compute_OP(u, opc.DOPE2, resname = 'DOPE')
# CHYO = OrderParameters().compute_OP(u, tail_data.CHYO, resname = 'CHYO')
#np.savetxt('POPC1.dat', POPC1)
#np.savetxt('POPC2.dat', POPC2)
#np.savetxt('DOPE1.dat', DOPE1)
#np.savetxt('DOPE2.dat', DOPE2)
np.savetxt('tail_bottom_3far.dat',POPC1)

# p=np.array(POPC1)
# p.shape
# p_plot = pd.DataFrame(p, columns=['Scd',
#                                   'Smectic'])
# p_plot.head()


# p_plot.plot(x='Scd')
# plt.title('Tail Order Parameters (Scd) (bottom close)')
# plt.ylabel('Scd') and plt.xlabel('Carbon')
# plt.savefig('order_popc_bottom_close.png')

# plt.draw()

# print('***Success***')
