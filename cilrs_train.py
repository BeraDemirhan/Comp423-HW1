import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt

from expert_dataset import ExpertDataset
from models.cilrs import CILRS

def validate(model, dataloader):
    """Validate model performance on the validation dataset"""
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for imgs, speeds, commands, actions in dataloader:
            # Move data to device
            imgs = imgs.to(device)
            speeds = speeds.to(device)
            commands = commands.to(device)
            actions = actions.to(device)

            # Forward pass
            pred_actions, pred_speeds = model(imgs, speeds, commands)
            loss = F.l1_loss(pred_actions, actions) + F.l1_loss(pred_speeds, speeds)

            total_loss += loss.item()

    return total_loss / len(dataloader)



def train(model, dataloader, optimizer):
    """Train model on the training dataset for one epoch"""
    model.train()
    total_loss = 0
    for imgs, speeds, commands, actions in dataloader:
        # Move data to device
        imgs = imgs.to(device)
        speeds = speeds.to(device)
        commands = commands.to(device)
        actions = actions.to(device)

        # Forward pass
        pred_actions, pred_speeds = model(imgs, speeds, commands)
        loss = F.l1_loss(pred_actions, actions) + F.l1_loss(pred_speeds, speeds)

        # Backward pass and optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)


def plot_losses(train_loss, val_loss):
    """Visualize the training and validation losses"""
    plt.plot(train_loss, label='Train Loss')
    plt.plot(val_loss, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    plt.savefig('losses.png')

def main():
    train_root = "/Users/berademirhan/Desktop/Comp423/hw1/expert_data/train"
    val_root = "/Users/berademirhan/Desktop/Comp423/hw1/expert_data/val"
    model = CILRS()
    train_dataset = ExpertDataset(train_root, task='imitation_learning')
    val_dataset = ExpertDataset(val_root, task='imitation_learning')

    # You can change these hyper parameters freely, and you can add more
    num_epochs = 10
    batch_size = 64
    save_path = "cilrs_model.ckpt"
    lr = 0.001

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_losses = []
    val_losses = []
    for i in range(num_epochs):
        train_loss = train(model, train_loader, optimizer)
        train_losses.append(train_loss)
        print(f"Epoch {i+1} train loss: {train_loss:.3f}")

        val_loss = validate(model, val_loader)
        val_losses.append(val_loss)
        print(f"Epoch {i+1} val loss: {val_loss:.3f}")

    torch.save(model.state_dict(), save_path)
    plot_losses(train_losses, val_losses)


if __name__ == "__main__":
    main()
