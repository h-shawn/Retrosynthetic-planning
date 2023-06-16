import pickle
import numpy as np
import pandas as pd
from rdkit.Chem import AllChem, Descriptors, MolFromSmiles, MolToSmiles
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


class TaskDataLoader:
    def __init__(self, path):
        self.path = path

    def load_property_data(self):
        with open(self.path + 'train.pkl', 'rb') as f:
            train = pickle.load(f)
        with open(self.path + 'test.pkl', 'rb') as f:
            test = pickle.load(f)
        X_train = np.unpackbits(train['packed_fp'][:80], axis=1)
        X_test = np.unpackbits(test['packed_fp'][:20], axis=1)

        Y_train = train['values'][:80].numpy()
        Y_test = test['values'][:20].numpy()
        return X_train, X_test, Y_train, Y_test


def transform_data(X_train, y_train, X_test, y_test, n_components=None, use_pca=False):
    x_scaler = StandardScaler()
    X_train_scaled = x_scaler.fit_transform(X_train)
    X_test_scaled = x_scaler.transform(X_test)
    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train)
    y_test_scaled = y_scaler.transform(y_test)

    if use_pca:
        pca = PCA(n_components)
        X_train_scaled = pca.fit_transform(X_train)
        print('Fraction of variance retained is: ' +
              str(sum(pca.explained_variance_ratio_)))
        X_test_scaled = pca.transform(X_test)

    return X_train_scaled, y_train_scaled, X_test_scaled, y_test_scaled, y_scaler


def featurise_mols(smiles_list, representation, bond_radius=3, nBits=2048):
    if representation == 'fingerprints':

        rdkit_mols = [MolFromSmiles(smiles) for smiles in smiles_list]
        X = [AllChem.GetMorganFingerprintAsBitVect(
            mol, bond_radius, nBits=nBits) for mol in rdkit_mols]
        X = np.asarray(X)

    elif representation == 'fragments':
        fragments = {d[0]: d[1] for d in Descriptors.descList[115:]}
        X = np.zeros((len(smiles_list), len(fragments)))
        for i in range(len(smiles_list)):
            mol = MolFromSmiles(smiles_list[i])
            try:
                features = [fragments[d](mol) for d in fragments]
            except:
                raise Exception('molecule {}'.format(
                    i) + ' is not canonicalised')
            X[i, :] = features

    else:
        # fragprints
        # convert to mol and back to smiles in order to make non-isomeric.
        rdkit_mols = [MolFromSmiles(smiles) for smiles in smiles_list]
        rdkit_smiles = [MolToSmiles(mol, isomericSmiles=False)
                        for mol in rdkit_mols]
        rdkit_mols = [MolFromSmiles(smiles) for smiles in rdkit_smiles]
        X = [AllChem.GetMorganFingerprintAsBitVect(
            mol, 3, nBits=2048) for mol in rdkit_mols]
        X = np.asarray(X)

        fragments = {d[0]: d[1] for d in Descriptors.descList[115:]}
        X1 = np.zeros((len(smiles_list), len(fragments)))
        for i in range(len(smiles_list)):
            mol = MolFromSmiles(smiles_list[i])
            try:
                features = [fragments[d](mol) for d in fragments]
            except:
                raise Exception('molecule {}'.format(
                    i) + ' is not canonicalised')
            X1[i, :] = features

        X = np.concatenate((X, X1), axis=1)

    return X
