import os
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
from monai.losses import DiceFocalLoss

import numpy as np
from PIL import Image
from torch import nn
from math import exp
from transformers import get_cosine_schedule_with_warmup

from args import get_train_args
from data import SalienciesDataset
from model import SaliencyFusionModel

def count_parameters(model):
  return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train(train_loader, validation_loader, saving_folder, num_epochs, n_layers):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = SaliencyFusionModel(
    in_saliencies=4 if "parasites" in str(saving_folder) else 3,
    in_channels=3 if "parasites" in str(saving_folder) else 1
  ).to(device)

  print("Number of parameters for :", count_parameters(model))

  criterion = DiceFocalLoss(
    include_background=False,
    to_onehot_y=False,
    sigmoid=False,
    lambda_focal=1.0,
    lambda_dice=1.0,
    gamma=1.0
  )

  l1_lambda = 1e-3

  optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
  num_steps = num_epochs * len(train_loader)
  warmup_steps = len(train_loader)
  scheduler = get_cosine_schedule_with_warmup(
      optimizer,
      num_warmup_steps=warmup_steps,
      num_training_steps=num_steps
  )
  best_val_loss = float('inf')


  for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    print("[INFO] Running Train")

    if epoch == 0:
      for param_group in optimizer.param_groups:
          param_group['lr'] = optimizer.param_groups[0]['lr'] * 0.1
    for batch_idx, (orig, sals, targets) in enumerate(train_loader):
      orig = orig.to(device)
      sals = sals.to(device)
      targets = targets.to(device)

      optimizer.zero_grad()

      outputs = model(orig, sals)
      loss = criterion(outputs, targets.unsqueeze(1))
      l1_reg = torch.tensor(0., requires_grad=True)
      for param in model.parameters():
        l1_reg = l1_reg + torch.norm(param, 1)
      loss = loss + l1_lambda * l1_reg

      loss.backward()
      optimizer.step()
      scheduler.step()

      train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    model.eval()
    val_loss = 0
    print("[INFO] Running Validation")
    with torch.no_grad():
      for orig, sals, targets in validation_loader:
        orig = orig.to(device)
        sals = sals.to(device)
        targets = targets.to(device)

        outputs = model(orig, sals)
        loss = criterion(outputs, targets.unsqueeze(1))
        l1_reg = torch.tensor(0., requires_grad=True)
        for param in model.parameters():
          l1_reg = l1_reg + torch.norm(param, 1)
        loss = loss + l1_lambda * l1_reg

        val_loss += loss.item()

    avg_val_loss = val_loss / len(validation_loader)

    print(f'Epoch [{epoch+1}/{num_epochs}]')
    print(f'Training Loss: {avg_train_loss:.4f}')
    print(f'Validation Loss: {avg_val_loss:.4f}')

    if avg_val_loss < best_val_loss:
      best_val_loss = avg_val_loss
      save_path = saving_folder / Path(f"model_epoch_{epoch+1}_loss_{best_val_loss}.pth")
      print(save_path)
      torch.save(
        {
          'epoch': epoch,
          'model_state_dict': model.state_dict(),
          'optimizer_state_dict': optimizer.state_dict(),
          'scheduler_state_dict': scheduler.state_dict(),
          'best_val_loss': best_val_loss
        },
        save_path
      )
      print(f'Model saved at epoch {epoch+1} with validation loss: {best_val_loss:.4f}')


def train_model(
  exp_folder, torig_folder, tlabel_folder, vorig_folder, vlabel_folder,
  m_folder, e
):
  print(f"[INFO] Running training")

  training_images = os.listdir(torig_folder)
  # If validation images are provided, get bets metrics losses on training
  if vorig_folder is None:
    vorig_folder = torig_folder
    vlabel_folder = tlabel_folder
    validation_images = training_images
  else:
    validation_images = os.listdir(vorig_folder)

  # Gets dataloaders
  if "parasites" in str(exp_folder):
    img_size = 400
    n_layers = 4
  elif "brain" in str(exp_folder):
    img_size = 240
    n_layers = 3

  train_dataset = SalienciesDataset(
    saliencies_folder=exp_folder, orig_folder=torig_folder,
    label_folder=tlabel_folder, img_list=training_images,
    n_layers=n_layers, img_size=img_size, augment=True
  )
  train_loader = DataLoader(
    train_dataset, batch_size=2, shuffle=True,
    num_workers=1, pin_memory=True
  )
  validation_dataset = SalienciesDataset(
    saliencies_folder=exp_folder, orig_folder=vorig_folder,
    label_folder=vlabel_folder, img_list=validation_images,
    n_layers=n_layers, img_size=img_size, augment=False
  )
  validation_loader = DataLoader(
    validation_dataset, batch_size=16, shuffle=False,
    num_workers=4, pin_memory=True
  )

  train(
    train_loader=train_loader,
    validation_loader=validation_loader,
    saving_folder=m_folder,
    num_epochs=e,
    n_layers=n_layers
  )

"""
  python3.11 train_merge_mode.py \
    -or /workdir/miscellaneous/parasites/split1/train_samples/orig/ \
    -lf /workdir/miscellaneous/parasites/split1/train_samples/label/ \
    -ef /workdir/out/parasites/split1/sample_ca_saliencies/ \
    -mf /workdir/out/parasites/split1/sample_merged_model \
    -e 2000 -vor None -vlf None

"""

if __name__ == "__main__":
  args = get_train_args()
  train_orig_folder = args.orig_folder
  train_label_folder = args.label_folder
  experiment_folder = args.experiment_folder
  m_folder = args.m_folder
  epochs = args.epochs
  val_orig_folder = args.val_orig_folder
  val_label_folder = args.val_label_folder

  os.makedirs(m_folder, exist_ok=True)
  train_model(
    experiment_folder, train_orig_folder, train_label_folder,
    val_orig_folder, val_label_folder, m_folder, epochs
  )