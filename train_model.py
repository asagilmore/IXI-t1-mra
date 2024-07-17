import niftitorch as nt
from os.path import join as pjoin
import torchvision.transforms.v2 as v2
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import argparse
import logging
import os

if __name__ == "__main__":
    # setup parser
    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('--data_dir', type=str, help='Path to the data '
                        'directory')
    parser.add_argument('--run_name', type=str,
                        help='Path to save model checkpoint ',
                        default='training_run')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_dir = args.data_dir
    train_dir = pjoin(data_dir, "train")
    val_dir = pjoin(data_dir, "valid")

    checkpoint_path = args.run_name + ".pth"

    # def transforms
    train_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32),
        nt.transforms.RandomRotation90(),
    ])

    train_dataset = nt.NiftiDataset(pjoin(train_dir, "images"),
                                    pjoin(train_dir, "masks"),
                                    train_transform)
    val_dataset = nt.NiftiDataset(pjoin(val_dir, "images"),
                                  pjoin(val_dir, "masks"),
                                  train_transform)

    # setup dataloaders
    train_loader = DataLoader(train_dataset, batch_size=20, shuffle=True,
                              num_workers=10, prefetch_factor=4,
                              pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=20, shuffle=False,
                            num_workers=10, prefetch_factor=4,
                            pin_memory=True)

    logging.basicConfig(filename=args.run_name + '.log', level=logging.INFO,
                        format='%(asctime)s:%(levelname)s:%(message)s')
    # setup combined perceptual loss with heavy bias toward MSE
    criterion = nt.losses.CombinedLoss(alpha=0.9, beta=0.1).to(device)
    # setup model
    model = nt.models.UNet(in_channels=1, out_channels=1,
                           ini_numb_filters=32).to(device)

    def init_weights(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    model.apply(init_weights)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                                                optimizer,
                                                mode='min',
                                                patience=5)

    if os.path.exists(checkpoint_path):
        print("model checkpoint found, loading model")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"]
    else:
        start_epoch = 0

    for epoch in range(start_epoch, 300):
        model.train()
        train_loss = 0.0
        total_samples = 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

        train_loss /= total_samples

        model.eval()
        val_loss = 0.0
        total_samples = 0
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            val_loss += loss.item() * inputs.size(0)
            total_samples += inputs.size(0)

        val_loss /= total_samples

        scheduler.step(val_loss)

        logging.info(f"Epoch: {epoch}, Train Loss: {train_loss}, "
                     f"Val Loss: {val_loss},")

        torch.save({
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch": epoch+1,
        }, checkpoint_path)
