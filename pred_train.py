import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from expert_dataset import ExpertDataset
from models.affordance_predictor import AffordancePredictor


def validate(model, dataloader):
    """Validate model performance on the validation dataset"""
    # Your code here
    model = model.eval()
    with torch.no_grad():
        for i, (img, affordance) in enumerate(dataloader):
            pred = model(img)
            loss = torch.nn.MSELoss(pred, affordance)
            print(loss)


def train(model, dataloader):
    """Train model on the training dataset for one epoch"""
    # Your code here
    model = model.train()
    for i, (img, affordance) in enumerate(dataloader):
        pred = model(img)
        loss = torch.nn.MSELoss(pred, affordance)
        print(loss)


def plot_losses(train_loss, val_loss):
    """Visualize your plots and save them for your report."""
    # Your code here
    train_loss = torch.tensor(train_loss)
    val_loss = torch.tensor(val_loss)
    plt.plot(train_loss, label='train')
    plt.plot(val_loss, label='val')
    plt.legend()
    plt.show()


def main():
    # Change these paths to the correct paths in your downloaded expert dataset
    train_root = "/Users/berademirhan/Desktop/Comp423/hw1/expert_data/train"
    val_root = "/Users/berademirhan/Desktop/Comp423/hw1/expert_data/val"
    model = AffordancePredictor()
    train_dataset = ExpertDataset(train_root)
    val_dataset = ExpertDataset(val_root)

    # You can change these hyper parameters freely, and you can add more
    num_epochs = 10
    batch_size = 64
    save_path = "pred_model.ckpt"

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    train_losses = []
    val_losses = []
    for i in range(num_epochs):
        train_losses.append(train(model, train_loader))
        val_losses.append(validate(model, val_loader))
    torch.save(model, save_path)
    plot_losses(train_losses, val_losses)


if __name__ == "__main__":
    main()
