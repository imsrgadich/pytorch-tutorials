# MNIST
# Data loader, Transformations
# Multilayer Neural Net, activation functions
# Loss and Optimizer
# Training loop (batch training)
# Model evaluation
# GPU support

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


class FeedForwardNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(FeedForwardNN, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        # no softmax here
        return out


def run_feedforward_nn_classification() -> None:
    # device config
    device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    # hyper parameters
    input_size = 784  # 28 x 28
    hidden_layer_size = 100
    num_classes = 10
    num_epoch = 2
    batch_size = 100
    learning_rate = 0.001

    # get the dataset
    train_dataset = torchvision.datasets.MNIST(root='./data',
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)

    test_dataset = torchvision.datasets.MNIST(root='./data',
                                               train=False,
                                               transform=transforms.ToTensor(),
                                               download=True)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                              batch_size=batch_size,
                                              shuffle=False)

    # model definition
    model = FeedForwardNN(input_size=input_size,
                          hidden_size=hidden_layer_size,
                          num_classes=num_classes)

    # loss criterion and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(),
                                 lr=learning_rate)
    # train loop
    num_total_steps = len(train_loader)
    for epoch in range(num_epoch):
        for i, (images, labels) in enumerate(train_loader):
            # images size 100, 1, 28,28 --> 100, 28x28
            images = images.reshape(-1, 28*28).to(device)
            labels = labels.to(device)

            # forward
            output = model(images)
            loss = criterion(output, labels)

            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 100 == 0:
                print(f"epoch: {epoch+1}/{num_epoch}, step: {i+1}/{num_total_steps}, loss: {loss.item():.4f}")

        # evaluate
        with torch.no_grad():
            num_correct = 0
            num_samples = 0
            for images, labels in test_loader:
                # images size 100, 1, 28,28 --> 100, 28x28
                images = images.reshape(-1, 28 * 28).to(device)
                labels = labels.to(device)

                # predictions
                outputs = model(images)
                _, predictions = torch.max(outputs, 1)
                num_samples += labels.shape[0]
                num_correct += (predictions == labels).sum().item()

            acc = 100.0 * num_correct/num_samples
            print(f"accuracy: {acc:.4f}")

    return None


if __name__ == "__main__":
    run_feedforward_nn_classification()
