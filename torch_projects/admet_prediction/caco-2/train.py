from data import train_loader, test_loader, valid_loader
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.nn import GCNConv, global_mean_pool
from tqdm import tqdm

class GNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim) -> None:
        super(GNN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_idx, batch = data.x, data.edge_index, data.batch
        x = torch.relu(self.conv1(x, edge_idx))
        x = torch.relu(self.conv2(x, edge_idx))
        x = torch.relu(self.conv3(x, edge_idx))
        x = global_mean_pool(x, batch)
        return self.fc(x)

def train_step(model, ):
    pass

def main():
    model = GNN(input_dim=9, hidden_dim=64, output_dim=1)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    num_epochs = 100
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss/len(train_loader):.4f}")

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in valid_loader:
                out = model(batch)
                val_loss += criterion(out, batch.y).item()
        print(f"Validation Loss: {val_loss/len(valid_loader):.4f}")


    # Test
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch in test_loader:
            out = model(batch)
            test_loss += criterion(out, batch.y).item()
    print(f"Test Loss: {test_loss/len(test_loader):.4f}")

    torch.save(model, "models/caco2_model.pt")

if __name__ == "__main__":
    main()
