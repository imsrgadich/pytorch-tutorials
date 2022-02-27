import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np


class ConvolutionalNN(nn.Module):
    def __init__(self):
        super(ConvolutionalNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=6, kernel_size=(5, 5))
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5, 5))
        self.fc1 = nn.Linear(in_features=16 * 5 * 5, out_features=120)
        self.fc2 = nn.Linear(in_features=120, out_features=84)
        self.fc3 = nn.Linear(in_features=84, out_features=10)

    def forward(self, x):
        # n x 3 x 32 x 32
        x = self.pool(F.relu(self.conv1(x)))  # n x 6 x 14 x 14
        x = self.pool(F.relu(self.conv2(x)))  # n x 16 x 5 x 5
        x = x.view(-1, 16 * 5 * 5)            # n x 400
        x = F.relu(self.fc1(x))               # n x 120
        x = F.relu(self.fc2(x))               # n x 84
        x = self.fc3(x)                       # n x 10
        return x


def run_cnn_classification() -> None:
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # hyper parameters
    num_epochs = 10
    batch_size = 4
    learning_rate = 0.01

    # dataset has PILImage images of range [0, 1].
    # We transform them to Tensors of normalized range [-1, 1]
    # https://stackoverflow.com/q/65467621/5483914
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=(0.5, 0.5, 0.5),
                              std=(0.5, 0.5, 0.5))]
    )

    # CIFAR10: 60000 32x32 color images in 10 classes, with 6000 images per class
    train_dataset = torchvision.datasets.CIFAR10(root="./data",
                                                 train=True,
                                                 download=True,
                                                 transform=transform)

    test_dataset = torchvision.datasets.CIFAR10(root="./data",
                                                train=False,
                                                download=True,
                                                transform=transform)

    train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                               batch_size=batch_size,
                                               shuffle=True)

    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                               batch_size=batch_size,
                                               shuffle=False)

    classes = ('plane', 'car', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck')

    """
    def imshow(img):
        img = img / 2 + 0.5  # unnormalize
        np_img = img.numpy()
        plt.imshow(np.transpose(np_img, (1, 2, 0)))
        plt.show()
        
    # to get some random samples
    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    
    # to plot some images
    imshow(torchvision.utils.make_grid(images))
    """

    # model definition
    model = ConvolutionalNN().to(device)

    # loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(params=model.parameters(),
                                lr=learning_rate)

    # training loop
    num_total_steps = len(train_loader)
    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
            #  4 x 3 x 32 x 32  --> 4 x 3 x 1024
            # input layer: 3 input channels,
            #              6 output channels,
            #              5 kernel size
            images = images.to(device)
            labels = labels.to(device)

            # forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if (i+1) % 2000 == 0:
                print(f"epoch: {epoch}/{num_epochs}, step: {i+1}/{num_total_steps}, loss: {loss.item():.4f}")

    print("Finished training")
    PATH = './cnn.pth'
    torch.save(model.state_dict(), PATH)

    # evaluation
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        n_class_correct = [0 for i in range(10)]
        n_class_samples = [0 for i in range(10)]

        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predictions = torch.max(input=outputs, dim=1)  # max returns value, index
            n_samples += labels.size(0)
            n_correct += (predictions == labels).sum().item()

            for i in range(batch_size):
                label = labels[i]
                pred = predictions[i]
                if (label == pred):
                    n_class_correct[label] += 1
                n_class_samples[label] += 1

        acc = 100.0 * n_correct / n_samples
        print(f"accuracy: {acc:.4f}")

        for i in range(10):
            acc = 100.0 * n_class_correct[i] / n_class_samples[i]
            print(f"accuracy of class {classes[i]}: {acc:.4f}")

    return None


if __name__ == "__main__":
    run_cnn_classification()
