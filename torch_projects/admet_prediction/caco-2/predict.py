import torch
from data import smiles_to_graph, test_loader, test_dataset
from train import GNN
import random
import argparse

def predict_from_smiles(model, smiles):
    graph = smiles_to_graph(smiles)
    if graph is None:
        print(f"Could not process SMILES: {smiles}")
        return None

    graph.batch = torch.zeros(graph.x.size(0), dtype=torch.long)

    model.eval()
    with torch.no_grad():
        prediction = model(graph)

    return prediction.item()

def main():
    parser = argparse.ArgumentParser(description="Predict Caco-2 permeability from test set")
    parser.add_argument("--model", type=str, default="models/caco2_model.pt", help="Path to the trained model")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to predict")
    args = parser.parse_args()

    # Load the model
    model = torch.load(args.model)

    # Select random samples from the test set
    num_samples = min(args.num_samples, len(test_dataset))
    sample_indices = random.sample(range(len(test_dataset)), num_samples)

    for idx in sample_indices:
        sample = test_dataset[idx]
        smiles = test_dataset.dataset.iloc[idx]['Drug']
        actual_value = sample.y.item()

        # Make prediction
        predicted_value = predict_from_smiles(model, smiles)

        if predicted_value is not None:
            print(f"SMILES: {smiles}")
            print(f"Actual Caco-2 permeability: {actual_value:.4f}")
            print(f"Predicted Caco-2 permeability: {predicted_value:.4f}")
            print(f"Absolute Error: {abs(actual_value - predicted_value):.4f}")
            print("--------------------")

if __name__ == "__main__":
    main()
