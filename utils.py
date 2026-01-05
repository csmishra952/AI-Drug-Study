import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from torch_geometric.data import Data

def smiles_to_graph(smiles):
    """
    Converts a SMILES string into a PyTorch Geometric Data object.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    xs = []
    for atom in mol.GetAtoms():
        xs.append([
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetFormalCharge(),
            atom.GetNumRadicalElectrons(),
            atom.GetHybridization(),
            atom.GetIsAromatic(),
            atom.GetTotalNumHs(),
            0, 
            0  
        ])
    x = torch.tensor(xs, dtype=torch.float)

    edge_indices = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_indices.append([i, j])
        edge_indices.append([j, i])  

    if not edge_indices:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()

    return Data(x=x, edge_index=edge_index)

def get_3d_block(smiles):
    """
    Generates a 3D text block for stmol visualization.
    """
    mol = Chem.MolFromSmiles(smiles)
    mol = Chem.AddHs(mol) 
    try:
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol) 
        return Chem.MolToMolBlock(mol)
    except:
        return None