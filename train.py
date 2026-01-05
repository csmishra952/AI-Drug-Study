import torch
from torch_geometric.datasets import MoleculeNet
from torch_geometric.loader import DataLoader
from model import GNN
import pandas as pd
print("Downloading Dataset...")
dataset = MoleculeNet(root='.', name='ESOL')
train_loader = DataLoader(dataset[:900], batch_size=64, shuffle=True)
test_loader = DataLoader(dataset[900:], batch_size=64, shuffle=False)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GNN(hidden_channels=64, num_node_features=dataset.num_node_features).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.MSELoss()
print("Starting Training...")
history = []
model.train()
for epoch in range(1, 101): 
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data.x.float(), data.edge_index, data.batch)
        loss = criterion(out, data.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch}: Loss {avg_loss:.4f}')
    history.append({'epoch': epoch, 'loss': avg_loss})
torch.save(model.state_dict(), 'drug_discovery_model.pt')
pd.DataFrame(history).to_csv('train_history.csv', index=False)
print("Model saved as 'drug_discovery_model.pt'!")