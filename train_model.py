import niftitorch as nt
from os.path import join as pjoin
import torchvision.transforms.v2 as v2
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import argparse


if __name__ == "__main__":
    # setup parser
    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('--data_dir', type=str, help='Path to the data '
                        'directory')
    args = parser.parse_args()

    data_dir = args.data_dir
    train_dir = pjoin(data_dir, "train")
    val_dir = pjoin(data_dir, "valid")

    # def transforms
    train_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32),
        nt.transforms.RandomRotation90(),
    ])

    train_dataset = nt.NiftiDataset(pjoin(train_dir, "images"),
                                    pjoin(train_dir, "mask"),
                                    train_transform)
    val_dataset = nt.NiftiDataset(pjoin(val_dir, "images"),
                                  pjoin(val_dir, "masks"),
                                  train_transform)

    # setup dataloaders
    train_loader = DataLoader(train_dataset, batch_size=50, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=50, shuffle=False)

    # setup combined perceptual loss with heavy bias toward MSE
    loss = nt.losses.CombinedLoss(alpha=0.9, beta=0.1)
    # setup model
    model = nt.models.UNet(in_channels=1, out_channels=1, ini_numb_filters=32)

    def init_weights(m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    model.apply(init_weights)

    model.train_unet(train_loader, val_loader, num_epochs=300, criterion=loss)
