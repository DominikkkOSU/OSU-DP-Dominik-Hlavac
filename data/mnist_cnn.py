import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import wandb

from generated_model import Net

def train(model, device, train_loader, optimizer, epoch):
    """
    Training loop for a single epoch. Modifies models weights and computes average loss & accuracy.
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward propergation
        output = model(data)
        loss = F.nll_loss(output, target)
        
        # Backward propagation and optimization
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * data.size(0)
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        total += data.size(0)

    # Calculate epoch metrics
    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total

    print(f'Train Epoch: {epoch}\tLoss: {epoch_loss:.6f}\tAccuracy: {epoch_acc:.2f}%')
    return epoch_loss, epoch_acc

def test(model, device, test_loader):
    """
    Testing loop. Evaluates the model on test data without modifying weights.
    """
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # Sum up batch loss
            test_loss += F.nll_loss(output, target, reduction='sum').item() 
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += data.size(0)

    test_loss /= total
    test_acc = 100. * correct / total

    print(f'Test set   : Average loss: {test_loss:.4f}, Accuracy: {test_acc:.2f}%\n')
    return test_loss, test_acc

def main():
    # 1. Initialize Weights & Biases
    wandb.init(project="nas-mnist-project")

    # 2. Define and Store Hyperparameters in wandb.config
    config = wandb.config
    config.learning_rate = 1.0
    config.batch_size = 64
    config.test_batch_size = 1000
    config.epochs = 5
    config.gamma = 0.7  # Learning rate step scheduler
    config.seed = 42
    config.network_architecture = "CNN (2 Conv, MaxPool, 2 FC, Dropout)"

    # Make runs reproducible
    torch.manual_seed(config.seed)

    # 3. Device Management (CUDA detection for headless servers)
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device execution context: {device}")

    # Setup DataLoader arguments depending on whether we use GPU
    train_kwargs = {'batch_size': config.batch_size, 'shuffle': True}
    test_kwargs = {'batch_size': config.test_batch_size, 'shuffle': False}
    
    if use_cuda:
        cuda_kwargs = {'num_workers': 2, 'pin_memory': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # Image transformation: convert to tensor and normalize based on known MNIST mean/std
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    # 4. Data Loading using torchvision
    print("Downloading and loading MNIST dataset...")
    # Downloads locally into data folder relative to cwd
    dataset1 = datasets.MNIST('./data', train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST('./data', train=False, download=True, transform=transform)
                       
    train_loader = DataLoader(dataset1, **train_kwargs)
    test_loader = DataLoader(dataset2, **test_kwargs)

    # 5. Initialize the Model, Optimizer, and LR Scheduler
    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=config.gamma)

    # 6. Main Training Loop
    final_test_acc = 0.0
    for epoch in range(1, config.epochs + 1):
        train_loss, train_acc = train(model, device, train_loader, optimizer, epoch)
        test_loss, test_acc = test(model, device, test_loader)
        
        final_test_acc = test_acc
        
        # Log metrics to wandb after every epoch
        wandb.log({
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_acc,
            "test_loss": test_loss,
            "test_accuracy": test_acc
        })

        # Step the learning rate scheduler
        scheduler.step()

    # 7. Log Final Summary Metric
    wandb.summary["final_test_accuracy"] = final_test_acc

    # 8. Model Saving
    save_path = "mnist_model.pth"
    torch.save(model.state_dict(), save_path)
    print(f"Model saved locally to {save_path}")
    
    # Close wandb context explicitly
    wandb.finish()

if __name__ == '__main__':
    main()
