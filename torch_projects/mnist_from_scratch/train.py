from activation import ReLU
from layers import Convolutional, Reshape
from loss import cross_entropy_loss
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.optim as optim

class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)

        # Max pooling layers
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        # First convolutional block
        x = self.pool(F.relu(self.conv1(x)))

        # Second convolutional block
        x = self.pool(F.relu(self.conv2(x)))

        # Flatten for fully connected layers
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


def train_step(model, x, y, learning_rate):
    # Ensure model is in training mode
    model.train()

    # Forward pass
    logits = model(x)

    # Compute loss
    loss = cross_entropy_loss(logits, y)

    # PyTorch's autograd system handles the backward pass
    # We don't need to manually call model.backward() as in your JAX version
    # Instead, we'll use PyTorch's optimizer for parameter updates

    # Zero the gradients
    model.zero_grad()

    # Backward pass
    loss.backward()

    # Update model parameters
    with torch.no_grad():
        for param in model.parameters():
            param -= learning_rate * param.grad

    return loss.item()  # Return the loss as a Python number


def main():
    if torch.backends.mps.is_available():
        mps_device = torch.device("mps")
        x = torch.ones(1, device=mps_device)
        print (x)
    else:
        print ("MPS device not found.")
        return

    img_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_data = datasets.MNIST(root='data', train=True, download=True, transform=img_transforms)
    test_data = datasets.MNIST(root='data', train=False, download=True, transform=img_transforms)

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=True)

    train_length = len(train_data)
    test_length = len(test_data)

    print(f"Number of training samples: {train_length}")
    print(f"Number of test samples: {test_length}")

    x_train , y_train = next(iter(train_loader))

    x_test , y_test = next(iter(test_loader))

    model = CustomModel()
    optimizer = 0.001  # Using SGD optimizer for simplicity
    num_epochs = 1000

    # Convert data to PyTorch tensors if not already
    x_train = torch.FloatTensor(x_train)
    y_train = torch.LongTensor(y_train)  # LongTensor for class indices in cross_entropy_loss

    for epoch in range(num_epochs):
        total_loss = 0
        for i in range(0, len(x_train), 64):  # Assuming batch size of 64
            batch_x = x_train[i:i+64]
            batch_y = y_train[i:i+64]

            # PyTorch's cross_entropy_loss expects class indices, not one-hot encoded
            # So we don't need to one-hot encode y_train

            loss = train_step(model, batch_x, batch_y, optimizer)
            total_loss += loss

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss / (len(x_train) / 64)}")

    # Plotting a sample prediction
    # model.eval()  # Set model to evaluation mode
    # with torch.no_grad():  # Disable gradient computation for inference
    #     sample_image = x_train[0:1]  # First image for simplicity
    #     sample_label = y_train[0:1]
    #     logits = model(sample_image)
    #     prediction = torch.argmax(logits, dim=1)
    #     true_label = sample_label

    # # Assuming x_train is normalized to [0, 1] or similar for visualization
    # plt.imshow(sample_image[0, 0].cpu().numpy(), cmap='gray')  # Assuming channel-first format
    # plt.title(f"Predicted: {prediction.item()}, True: {true_label.item()}")
    # plt.show()


if __name__ == "__main__":
    main()
