import niftitorch as nt
from os.path import join as pjoin
import torchvision.transforms.v2 as v2
import torch
from torch.utils.data import DataLoader
import argparse
import logging
from train_logger import Logger
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

    logger = Logger(args.run_name + '.csv')
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
    val_loader = DataLoader(val_dataset, batch_size=20, shuffle=False,
                            num_workers=args.cpus, prefetch_factor=4,
                            pin_memory=True)

    logging.basicConfig(filename=args.run_name + '.log', level=logging.INFO,
                        format='%(asctime)s:%(levelname)s:%(message)s')

    model = nt.models.UNet(1, 1).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                                                optimizer,
                                                mode='min',
                                                patience=5)

    criterion = nt.losses.CombinedLoss(alpha=0.9, beta=0.1).to(device)

    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
    else:
        start_epoch = 0

    for epoch in range(start_epoch, args.epochs):
        train_loss = model.train_self(train_loader, optimizer, criterion)
        val_loss = model.valid_self(val_loader, criterion)

        scheduler.step(val_loss)

        logger.log_epoch(epoch, {'train_loss': train_loss,
                                 'val_loss': val_loss,
                                 'lr': scheduler.get_last_lr()})
        logging.info(f"Epoch {epoch} train_loss: {train_loss} val_loss: "
                     f"{val_loss} lr: {scheduler.get_last_lr()}")

        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'epoch': epoch}, checkpoint_path)
