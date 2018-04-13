"""
Script to access molecules from CHEMBL via their API

"""
import os
from chembl_webresource_client.new_client import new_client
from joblib import Parallel, delayed
import multiprocessing

# code from CHEMBL documentation
molecule = new_client.molecule

light_molecules = molecule.filter(molecule_properties__mw_freebase__lte=200)
# ######## end CHEMBL code

def get_data(mol):
    # # begin code to make data files
    # for mol in range(len(light_molecules)):
        # # only get molecules that contain a valid smiles (for example cisplatin does not have one but it is listed as a drug for lung cancer)
    try:
        # get molecule information
        smiles = light_molecules[mol]['molecule_structures']['canonical_smiles']
        # use chembl_id to get the image of the molecule
        chembl_id = light_molecules[mol]['molecule_chembl_id']

        # get and save molecule image as png
        img = new_client.image.get(chembl_id)
        with open('{}.png'.format(chembl_id), 'wb') as f:
            f.write(img)

        # store everything in a directory
        # store smiles to a txt file
        with open('{}_smiles.txt'.format(chembl_id), 'w') as f:
            f.write(smiles)

        # use pref_name of molecule as directory name
        os.system('mkdir -p mol_wt_mols/' + chembl_id)
        os.system('mv {}_smiles.txt {}.png mol_wt_mols/{}'.format(chembl_id,chembl_id, chembl_id))

        print('Finished {}'.format(chembl_id))
    # if the molecule doesn't have a canonical smiles field
    except TypeError:
        pass
num_cores = multiprocessing.cpu_count()
Parallel(n_jobs = num_cores, verbose=50)(delayed(get_data)(mol) for mol in range(len(light_molecules)))
