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
    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('--data_dir', type=str, help='Path to the data '
                        'directory')
    parser.add_argument('--run_name', type=str,
                        help='Path to save model checkpoint ',
                        default='training_run')
    parser.add_argument('--cpus', type=int, help='Number of cpus to use',
                        default=4)
    parser.add_argument('--epochs', type=int, help='Number of epochs to train',
                        default=100)

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
                              num_workers=args.cpus, prefetch_factor=4,
                              pin_memory=True)
    second_train_loader = DataLoader(train_dataset, batch_size=20,
                                     shuffle=True, num_workers=args.cpus,
                                     prefetch_factor=4, pin_memory=True)

    val_loader = DataLoader(val_dataset, batch_size=20, shuffle=False,
                            num_workers=args.cpus, prefetch_factor=4,
                            pin_memory=True)

    logging.basicConfig(filename=args.run_name + '.log', level=logging.INFO,
                        format='%(asctime)s:%(levelname)s:%(message)s')

    model = nt.models.WassersteinGAN().to(device)
    critic_optimizer = torch.optim.Adam(model.critic.parameters(), lr=1e-4)
    generator_optimizer = torch.optim.Adam(model.generator.parameters(),
                                           lr=1e-4)

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        critic_optimizer.load_state_dict(
                         checkpoint['critic_optimizer_state_dict'])
        generator_optimizer.load_state_dict(
                            checkpoint['generator_optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
    else:
        start_epoch = 0

    for epoch in range(start_epoch, args.epochs):
        loss_dict = model.train_self(train_loader, second_train_loader,
                                     critic_optimizer, generator_optimizer,
                                     device=device)
        logging.info(f"Train Loss, {epoch+1}/{args.epochs}: {loss_dict}")

        loss_dict = model.validate(val_loader, device=device)
        logging.info(f"Validation Loss, {epoch+1}/{args.epochs}: {loss_dict}")
