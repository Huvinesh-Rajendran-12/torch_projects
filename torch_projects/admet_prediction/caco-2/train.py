from data import train_loader, test_loader, valid_loader
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import argparse
import csv
from model import GCN, MPNN

def train_step(model, optimizer, criterion, batch, device):
    model.train()
    optimizer.zero_grad()
    batch = batch.to(device)
    out = model(batch)
    loss = criterion(out, batch.y)
    loss.backward()
    optimizer.step()
    return loss.item()

def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            total_loss += criterion(out, batch.y).item()
    return total_loss / len(loader)

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if args.model == 'GCN':
        model = GCN(input_dim=9, hidden_dim=args.hidden_dim, output_dim=1).to(device)
    elif args.model == 'MPNN':
        model = MPNN(input_dim=9, hidden_dim=args.hidden_dim, output_dim=1).to(device)
    else:
        raise ValueError("Invalid model name. Choose 'GCN' or 'MPNN'.")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    results = []
    best_val_loss = float('inf')

    for epoch in range(args.epochs):
        total_loss = 0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}"):
            loss = train_step(model, optimizer, criterion, batch, device)
            total_loss += loss

        train_loss = total_loss / len(train_loader)
        val_loss = evaluate(model, valid_loader, criterion, device)

        print(f"Epoch {epoch+1}/{args.epochs}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}")

        results.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'val_loss': val_loss
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model, f"models/{args.model}_best.pt")

    test_loss = evaluate(model, test_loader, criterion, device)
    print(f"Test Loss: {test_loss:.4f}")

    # Save results to CSV
    with open(f'{args.model}_results.csv', 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=['epoch', 'train_loss', 'val_loss'])
        writer.writeheader()
        writer.writerows(results)

    # Save final model
    torch.save(model, f"models/{args.model}_final.pt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a GNN model for Caco-2 prediction")
    parser.add_argument('--model', type=str, choices=['GCN', 'MPNN'], default='GCN', help='Model architecture to use')
    parser.add_argument('--hidden_dim', type=int, default=64, help='Hidden dimension of the model')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
    args = parser.parse_args()

    main(args)
