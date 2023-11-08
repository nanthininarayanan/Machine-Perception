#! /usr/bin/env python3

import torch
import argparse
import torchvision
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import numpy as np

from tqdm import tqdm
from models_list import get_model
from datetime import datetime
from sklearn.metrics import roc_curve, auc, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


def print_layers_and_params(model, printfile):
    for name, layer in model.named_children():
        num_params = sum(p.numel() for p in layer.parameters() if p.requires_grad)
        print(f"Layer name: {name}", file=printfile)
        print(f"Trainable Parameters: {num_params}", file=printfile)
        print(layer, file=printfile)


# Training the model
def train(net, train_loader, criterion, optimizer, num_epochs, device, printfile):
    train_losses = []
    train_accuracies = []

    print("Running on device:", device, file=printfile)

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        epoch_pbar = tqdm(
            enumerate(train_loader, 0),
            total=len(train_loader),
            desc=f"Epoch {epoch + 1}/{num_epochs}",
        )

        for i, data in epoch_pbar:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels).to(device)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            epoch_pbar.set_postfix(
                {
                    "Epoch Loss": running_loss / (i + 1),
                    "Epoch Accuracy": 100 * correct / total,
                }
            )

        train_losses.append(running_loss / len(train_loader))
        train_accuracies.append(100 * correct / total)

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    torch.save(net.state_dict(), f"./models/{args.model_name}_{now}.pth")
    # net.save(f"{args.model_name}_{now}.pth")
    return train_losses, train_accuracies


# Define a function to calculate additional metrics and plot training accuracy/loss
def evaluate_and_plot_metrics(
    net, test_loader, train_losses, train_accuracies, device, model_name, printfile
):
    net.eval()
    all_preds = []
    all_labels = []
    print("Evaluating on device:", device, file=printfile)
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate accuracy
    accuracy = accuracy_score(all_labels, all_preds)

    # Calculate confusion matrix
    confusion = confusion_matrix(all_labels, all_preds)

    # Calculate classification report with F1 score, precision, specificity, and sensitivity
    class_report = classification_report(
        all_labels,
        all_preds,
        target_names=[
            "Airplane",
            "Automobile",
            "Bird",
            "Cat",
            "Deer",
            "Dog",
            "Frog",
            "Horse",
            "Ship",
            "Truck",
        ],
        output_dict=True,
    )

    print(f"Test Accuracy: {accuracy * 100:.2f}%", file=printfile)

    print("Confusion Matrix:", file=printfile)
    print(confusion, file=printfile)
    disp = ConfusionMatrixDisplay(confusion_matrix=confusion)
    disp.plot()
    
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"./figures/confusion_{model_name}_{now}.png")

    print("Classification Report:", file=printfile)
    print(class_report, file=printfile)

    # Plot the training accuracy
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    print()
    plt.plot(train_losses)
    plt.xlabel("Epochs")
    plt.ylabel("Training Loss")
    plt.title("Training Loss Over Time")

    # Plot the training accuracy
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies)
    plt.xlabel("Epochs")
    plt.ylabel("Training Accuracy")
    plt.title("Training Accuracy Over Time")
    
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"./figures/train_{model_name}_{now}.png")

# Function to create an ROC curve for a specific class
def plot_roc_curve(net, test_loader, target_class, model_name, device):
    net.eval()
    all_scores = []
    all_labels = []
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = net(inputs)
            all_scores.extend(outputs[:, target_class].cpu().numpy())
            all_labels.extend((labels == target_class).cpu().numpy())

    fpr, tpr, thresholds = roc_curve(all_labels, all_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (AUC = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for Class {target_class}')
    plt.legend(loc='lower right')
    
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"./figures/roccurve_{model_name}_target{target_class}_{now}.png")


def main(args):
    device = torch.device(
        "mps"
        if torch.backends.mps.is_available()
        else "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    # device = torch.device("cpu")

    printfile = open(
        f"./output/{args.model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        "a",
    )

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(
            root="./data", train=True, download=True, transform=transform
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=transform
        ),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )

    net = get_model(args.model_name)
    print(
        "Layers and Trainable Parameters of",
        args.model_name,
        ":\n",
        file=printfile,
    )
    print_layers_and_params(net, printfile)

    net.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(net.parameters(), lr=args.lr)

    train_losses, train_accuracies = train(
        net, train_loader, criterion, optimizer, args.epochs, device, printfile
    )
    print(f"Train Losses: {train_losses}", file=printfile)
    print(f"Train Accuracies: {train_accuracies}", file=printfile)
    
    evaluate_and_plot_metrics(
        net,
        test_loader,
        train_losses,
        train_accuracies,
        device,
        args.model_name,
        printfile,
    )

    for target_class in range(10):
        plot_roc_curve(net,
            test_loader,
            target_class,
            args.model_name,
            device
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch CIFAR10 Training")

    parser.add_argument("--lr", default=0.001, type=float, help="learning rate")
    parser.add_argument("--epochs", default=20, type=int, help="number of epochs")
    parser.add_argument("--batch_size", default=32, type=int, help="batch size")
    parser.add_argument("--num_workers", default=2, type=int, help="number of workers")
    parser.add_argument(
        "--model_name",
        choices=["CNN", "CustomCNN", "TransferCNN"],
        help="model name",
    )

    args = parser.parse_args()

    main(args)
