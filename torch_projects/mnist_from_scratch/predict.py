import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from train import CustomModel
import matplotlib.pyplot as plt

def main():
    img_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    test_data = datasets.MNIST(root='data', train=False, download=True, transform=img_transforms)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=True)
    x_test , y_test = next(iter(test_loader))
    model = torch.load(f="models/mnist.pt")
    cmp_model = torch.compile(model)
    with torch.no_grad():
        sample_img = x_test[0:1]
        sample_label = y_test[0:1]
        logits = cmp_model(sample_img)
        pred = torch.argmax(logits, dim=1)

    plt.imshow(sample_img[0, 0].cpu().numpy(), cmap='gray')  # Assuming channel-first format
    plt.title(f"Predicted: {pred.item()}, True: {sample_label.item()}")
    plt.show()


if __name__ == "__main__":
    main()
