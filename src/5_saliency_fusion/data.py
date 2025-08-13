import os
from PIL import Image
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import torch
import random
import torchvision.transforms as T
import torchvision.transforms.functional as TF

class SynchronizedTransform:
  def __init__(self, img_size):
    self.img_size = img_size

    # Store parameters for synchronized transforms
    self.params = None

  def _get_transform_params(self, img):
    if self.params is None:
      # Initialize dictionary to store parameters
      self.params = {}

      # Store crop parameters
      i, j, h, w = T.RandomResizedCrop.get_params(
          img, scale=(0.9, 1.0), ratio=(1.0, 1.0))
      self.params['crop'] = (i, j, h, w)

      # Store flip parameters
      self.params['vflip'] = random.random() < 0.5
      self.params['hflip'] = random.random() < 0.5

      # Store rotation parameters
      self.params['rotation'] = random.uniform(-15, 15)

      # Store other parameters
      self.params['sharpness'] = random.random() < 0.5

      # Store perspective parameters
      w, h = img.size
      distortion_scale = 0.2
      half_height = h // 2
      half_width = w // 2
      topleft = [
          int(random.uniform(0, distortion_scale * half_width)),
          int(random.uniform(0, distortion_scale * half_height))
      ]
      topright = [
          int(random.uniform(w - distortion_scale * half_width, w)),
          int(random.uniform(0, distortion_scale * half_height))
      ]
      botright = [
          int(random.uniform(w - distortion_scale * half_width, w)),
          int(random.uniform(h - distortion_scale * half_height, h))
      ]
      botleft = [
          int(random.uniform(0, distortion_scale * half_width)),
          int(random.uniform(h - distortion_scale * half_height, h))
      ]
      self.params['perspective_endpoints'] = (topleft, topright, botright, botleft)
      self.params['perspective'] = random.random() < 0.5

      # Store affine parameters
      self.params['affine'] = {
          'angle': random.uniform(-30, 30),
          'translate': (random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1)),
          'scale': random.uniform(0.8, 1.2)
      }

      # Store color parameters
      self.params['brightness'] = random.uniform(0.8, 1.2)
      self.params['contrast'] = random.uniform(0.8, 1.2)

    return self.params

  def __call__(self, img, mask=False, rgb=False):
    params = self._get_transform_params(img)

    # Apply transforms with stored parameters
    img = TF.resized_crop(img, *params['crop'], self.img_size)
    if params['vflip']:
      img = TF.vflip(img)
    if params['hflip']:
      img = TF.hflip(img)

    img = TF.affine(img,
                    angle=params['affine']['angle'],
                    translate=params['affine']['translate'],
                    scale=params['affine']['scale'],
                    shear=0)
    w, h = img.size
    if params['perspective']:
      startpoints = [[0, 0], [w-1, 0], [w-1, h-1], [0, h-1]]
      img = TF.perspective(img, startpoints=startpoints,
                          endpoints=params['perspective_endpoints'])
    img = TF.rotate(img, params['rotation'])
    if not mask:
      if params['sharpness']:
        img = TF.adjust_sharpness(img, 2.0)

      img = TF.gaussian_blur(img, 3, sigma=(0.1, 2.0))
      img = TF.adjust_brightness(img, params['brightness'])
      img = TF.adjust_contrast(img, params['contrast'])

    img = TF.to_tensor(img)

    if not mask and not rgb:
      img = TF.normalize(img, mean=[0.5], std=[0.5])
    if rgb:
      img = TF.normalize(
        img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
      )

    return img

  def reset_parameters(self):
    """Reset stored parameters to generate new random transforms"""
    self.params = None

class SalienciesDataset(Dataset):
  def __init__(
    self, saliencies_folder, orig_folder, label_folder, img_list, n_layers,
    img_size, augment
  ):
    self.saliencies_folder = saliencies_folder
    self.orig_folder = orig_folder
    self.label_folder = label_folder
    self.img_list = img_list
    self.augment = augment
    self.n_layers = n_layers
    self.img_size = img_size
    self.augment = augment
    self.norm_value = 255 if img_size == 400 else 65535

    if self.augment:
      self.transform = SynchronizedTransform(self.img_size)
    else:
      self.transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5])
      ])
      self.orig_transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.5], std=[0.5])
      ])

    self.mask_transform = T.Compose([
      T.ToTensor()
    ])

  def __len__(self):
    return len(self.img_list)

  def __getitem__(self, idx):
    saliencies = []
    img_file = Path(f"{self.img_list[idx]}")

    label_path = self.label_folder / Path(str(img_file))
    label = Image.open(label_path).convert('L')
    label_shape = label.size
    orig_path = self.orig_folder / Path(str(img_file))
    orig_img = Image.open(orig_path)
    orig_img = np.array(orig_img)
    orig_img = orig_img.astype(np.float64) / self.norm_value
    orig_img = (orig_img * 255).astype(np.uint8)
    orig_img = Image.fromarray(orig_img)

    for nl in range(1, self.n_layers + 1):
      saliency_path = self.saliencies_folder / Path(f"layer_{nl}") / str(img_file).replace(".png", "_sal.png")
      saliency = Image.open(saliency_path).convert('L')
      saliency = saliency.resize(label_shape, Image.NEAREST)

      if self.augment:
        saliencies.append(self.transform(saliency, mask=True).squeeze(0))
      else:
        saliencies.append(self.mask_transform(saliency).squeeze(0))

    saliencies = torch.stack(saliencies, dim=0)

    if self.augment:
      label = self.transform(label, mask=True).squeeze(0)
      orig_img = self.transform(orig_img, rgb=True if self.norm_value==255 else False)
      self.transform.reset_parameters()
    else:
      label = self.mask_transform(label).squeeze(0)
      orig_img = self.orig_transform(orig_img)

    return orig_img, saliencies, label


class InferenceSalienciesDataset(Dataset):
  def __init__(
    self, orig_folder, saliencies_folder, img_list, n_layers,
    img_size
  ):
    self.orig_folder = orig_folder
    self.saliencies_folder = saliencies_folder
    self.img_list = img_list
    self.n_layers = n_layers
    self.img_size = img_size
    self.norm_value = 255 if img_size == 400 else 65535


    self.transform = T.Compose([
      T.ToTensor(),
      T.Normalize(mean=[0.5], std=[0.5])
    ])

    self.orig_transform = T.Compose([
      T.ToTensor(),
      T.Normalize(mean=[0.5], std=[0.5])
    ])

  def __len__(self):
    return len(self.img_list)

  def __getitem__(self, idx):
    saliencies = []
    self.img_file = Path(f"{self.img_list[idx]}.png")

    for nl in range(1, self.n_layers + 1):
      saliency_path = self.saliencies_folder / Path(f"layer_{nl}") / self.img_file
      saliency = Image.open(saliency_path).convert('L')
      saliency = saliency.resize(self.img_size, Image.NEAREST)

      saliencies.append(self.transform(saliency).squeeze(0))

    saliencies = torch.stack(saliencies, dim=0)

    orig_path = self.orig_folder / Path(str(self.img_file).replace("_sal", ""))
    orig_img = Image.open(orig_path)
    orig_img = np.array(orig_img)
    orig_img = orig_img.astype(np.float64) / self.norm_value
    orig_img = (orig_img * 255).astype(np.uint8)

    orig_img = Image.fromarray(orig_img)
    orig_img = self.orig_transform(orig_img)

    return orig_img, saliencies