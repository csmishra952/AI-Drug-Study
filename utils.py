import torch
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch_geometric.data import Data
from torch_geometric.explain import Explainer, GNNExplainer
ATOM_LIST = ['C', 'N', 'O', 'S', 'F', 'Cl', 'Br', 'I', 'P', 'Unknown']
HYBRIDIZATION_LIST = [
    Chem.rdchem.HybridizationType.SP,
    Chem.rdchem.HybridizationType.SP2,
    Chem.rdchem.HybridizationType.SP3,
    Chem.rdchem.HybridizationType.SP3D,
    Chem.rdchem.HybridizationType.SP3D2
]

def one_hot_encoding(value, choices):
    """
    Creates a one-hot vector (e.g., [0, 1, 0, 0]) for a value based on a list of choices.
    """
    encoding = [0] * (len(choices))
    index = choices.index(value) if value in choices else -1
    if index != -1:
        encoding[index] = 1
    return encoding

def get_atom_features(atom):
    """
    Extracts a robust feature vector for a single atom.
    Features: Atom type (one-hot), Degree (one-hot), Formal Charge, 
              Hybridization (one-hot), Aromaticity, Total H.
    """
    features = (
        one_hot_encoding(atom.GetSymbol(), ATOM_LIST) +
        one_hot_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5]) +
        [atom.GetFormalCharge()] +
        one_hot_encoding(atom.GetHybridization(), HYBRIDIZATION_LIST) +
        [1 if atom.GetIsAromatic() else 0] +
        [atom.GetTotalNumHs()]
    )
    return np.array(features, dtype=np.float32)

def smiles_to_graph(smiles):
    """
    Converts a SMILES string into a PyTorch Geometric Data object with rich features.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    xs = []
    for atom in mol.GetAtoms():
        xs.append(get_atom_features(atom))
    x = torch.tensor(np.array(xs), dtype=torch.float)

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

def scaffold_split(dataset, seed=42, frac=[0.8, 0.1, 0.1]):
    """
    Performs Scaffold Splitting: Grouping molecules by their core structure.
    This prevents the model from just memorizing similar molecules.
    """
    scaffolds = {}
    for i, data in enumerate(dataset):
        smiles = data.smiles 
        scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=Chem.MolFromSmiles(smiles), includeChirality=False)
        if scaffold not in scaffolds:
            scaffolds[scaffold] = []
        scaffolds[scaffold].append(i)
    
    scaffold_sets = [scaffolds[k] for k in sorted(scaffolds, key=lambda x: len(scaffolds[x]), reverse=True)]
    
    train_idx, val_idx, test_idx = [], [], []
    train_cutoff = len(dataset) * frac[0]
    val_cutoff = len(dataset) * (frac[0] + frac[1])
    
    current_count = 0
    for group in scaffold_sets:
        if current_count < train_cutoff:
            train_idx.extend(group)
        elif current_count < val_cutoff:
            val_idx.extend(group)
        else:
            test_idx.extend(group)
        current_count += len(group)
        
    return train_idx, val_idx, test_idx


def get_molecule_explanation(model, data):
    """
    Uses GNNExplainer to find which atoms define the property.
    Returns: A list of importance scores (0 to 1) for each atom.
    """
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=100), 
        explanation_type='model',
        node_mask_type='object',
        edge_mask_type='object',
        model_config=dict(
            mode='regression',
            task_level='graph',
            return_type='raw',
        ),
    )
    
    batch = torch.zeros(data.x.shape[0], dtype=torch.long, device=data.x.device)
    explanation = explainer(data.x, data.edge_index, batch=batch)
    
    node_mask = explanation.node_mask
    if node_mask is None:
        return np.zeros(data.x.shape[0])
    scores = node_mask.sum(dim=1).cpu().detach().numpy()
    if scores.max() > scores.min():
        scores = (scores - scores.min()) / (scores.max() - scores.min())
    else:
        scores = scores / (scores.max() + 1e-6)
        
    return scores

def get_colored_3d_block(smiles, importance_scores=None):
    """
    Generates a 3D block and color commands for py3Dmol based on importance.
    """
    mol = Chem.MolFromSmiles(smiles)
    if not mol: return None, None
    mol = Chem.AddHs(mol)
    
    try:
        AllChem.EmbedMolecule(mol, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(mol)
        block = Chem.MolToMolBlock(mol)
    except:
        return None, None
    style_spec = {}
    if importance_scores is not None:
        score_idx = 0
        
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() != 1: 
                if score_idx < len(importance_scores):
                    val = importance_scores[score_idx]
                    
                    if val < 0.5:
                        ratio = val * 2
                        r = int(255 * ratio)
                        g = int(255 * ratio)
                        b = 255
                    else:
                        ratio = (val - 0.5) * 2
                        r = 255
                        g = int(255 * (1 - ratio))
                        b = int(255 * (1 - ratio))
                    
                    color_hex = f'#{r:02x}{g:02x}{b:02x}'
                    style_spec[atom.GetIdx()] = color_hex
                    score_idx += 1
            else:
                style_spec[atom.GetIdx()] = '#ffffff' 

    return block, style_spec