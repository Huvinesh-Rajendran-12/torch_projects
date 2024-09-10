from re import A
from tdc.single_pred import ADME
from rdkit import Chem
from rdkit.Chem import Draw
from torch_geometric.data import Data, Dataset
from torch_geometric.loader import DataLoader
import torch
import networkx as nx
import matplotlib.pyplot as plt

# utility functions and classes

def smiles_to_graph(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    atom_features = []
    for atom in mol.GetAtoms():
        features = [
            atom.GetAtomicNum(),
            atom.GetDegree(),
            atom.GetFormalCharge(),
            atom.GetNumRadicalElectrons(),
            int(atom.GetHybridization()),
            int(atom.GetIsAromatic()),
            atom.GetMass(),
            atom.GetExplicitValence(),
            atom.GetImplicitValence()
        ]
        atom_features.append(features)

    edge_indices = []
    edge_features = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_indices.extend([[i, j], [j, i]])

        features = [
            bond.GetBondTypeAsDouble(),
            int(bond.GetIsConjugated()),
            int(bond.GetIsAromatic())
        ]
        edge_features.extend([features, features])

    x = torch.tensor(atom_features, dtype=torch.float)
    edge_idx = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    edge_attr = torch.tensor(edge_features, dtype=torch.float)

    return Data(x=x, edge_index=edge_idx, edge_attr=edge_attr)

def visualize_molecule_graph(smiles):
    # Convert SMILES to graph
    data = smiles_to_graph(smiles)

    if data is None:
        print(f"Could not process SMILES: {smiles}")
        return

    # Create networkx graph
    G = nx.Graph()

    # Add nodes
    for i in range(data.x.shape[0]):
        G.add_node(i)

    # Add edges
    edge_index = data.edge_index.t().tolist()
    for edge in edge_index:
        G.add_edge(edge[0], edge[1])

    # Set up plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Draw molecule
    mol = Chem.MolFromSmiles(smiles)
    img = Draw.MolToImage(mol)
    ax1.imshow(img)
    ax1.axis('off')
    ax1.set_title("Molecule")

    # Draw graph
    pos = nx.spring_layout(G)
    nx.draw(G, pos, ax=ax2, with_labels=True, node_color='lightblue',
            node_size=500, font_size=10, font_weight='bold')
    ax2.set_title("Graph Representation")

    plt.tight_layout()
    plt.show()


class DrugDataset(Dataset):
    def __init__(self, dataset, root=None, transform=None, pre_transform=None):
        self.dataset = dataset
        super().__init__(root, transform, pre_transform)

    def len(self):
        return len(self.dataset)

    def get(self, idx):
        row = self.dataset.iloc[idx]
        smiles = row['Drug']
        y = row['Y']

        graph = smiles_to_graph(smiles)
        if graph is None:
            # Return a dummy graph if conversion fails
            graph = Data(x=torch.tensor([[0]]), edge_index=torch.tensor([[0], [0]]))

        graph.y = torch.tensor([y], dtype=torch.float)
        return graph



# transform the data

data = ADME(name='Caco2_Wang')

split = data.get_split()

train_dataset = DrugDataset(split['train'])
test_dataset = DrugDataset(split['test'])
valid_dataset = DrugDataset(split['valid'])

# Create data loaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
valid_loader = DataLoader(valid_dataset, batch_size=64, shuffle=False)
