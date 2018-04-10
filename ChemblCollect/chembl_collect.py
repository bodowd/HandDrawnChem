import os
from chembl_webresource_client.new_client import new_client


drug_indication = new_client.drug_indication
molecules = new_client.molecule

lung_cancer_ind = drug_indication.filter(efo_term__icontains="LUNG CARCINOMA")
# returns list of dictionary of 216 molecules and data on the mols
lung_cancer_mols = molecules.filter(
    molecule_chembl_id__in=[x['molecule_chembl_id'] for x in lung_cancer_ind]
)

for mol in range(len(lung_cancer_mols)):
    # only get molecules that contain a valid smiles (for example cisplatin does not have one but it is listed as a drug for lung cancer)
    try:
        # get molecule information
        smiles = lung_cancer_mols[mol]['molecule_structures']['canonical_smiles']
        name = lung_cancer_mols[mol]['pref_name']
        # use chembl_id to get the image of the molecule
        chembl_id = lung_cancer_mols[mol]['molecule_chembl_id']

        # get and save molecule image as png
        img = new_client.image.get(chembl_id)
        with open('{}.png'.format(chembl_id), 'wb') as f:
            f.write(img)

        # store everything in a directory
        # store smiles to a txt file
        with open('smiles.txt', 'w') as f:
            f.write(smiles)

        # use pref_name of molecule as directory name
        os.system('mkdir -p data/' + name)
        os.system('mv smiles.txt {}.png data/{}'.format(chembl_id, name))

        print('Finished {}'.format(name))

    except TypeError:
        pass
