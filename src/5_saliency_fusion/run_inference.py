from pathlib import Path
import os
import re
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from PIL import Image

from args import get_inferece_args
from model import SaliencyFusionModel
from data import InferenceSalienciesDataset


def get_epoch_number(filename):
  match = re.search(r'epoch_(\d+)_', filename)
  return int(match.group(1)) if match else -1

def get_best_model(model_folder):
  weight_files = os.listdir(model_folder)
  sorted_files = sorted(weight_files, key=get_epoch_number, reverse=True)
  best_weight = sorted_files[0]
  model_path = model_folder / Path(best_weight)

  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  model = SaliencyFusionModel(
    in_saliencies=4 if "parasites" in str(model_folder) else 3,
    in_channels=3 if "parasites" in str(model_folder) else 1
  ).to(device)

  print(f"[INFO] Reading weight file: {model_path}")
  checkpoint = torch.load(model_path, map_location=torch.device(device), weights_only=True)
  model.load_state_dict(checkpoint['model_state_dict'])

  return model

def run_inference(model, dataset, saving_folder):
  for origs, saliencies in tqdm(iter(dataset)):
    img_file = dataset.img_file
    saving_path = saving_folder / Path(img_file)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    origs = origs.to(device)
    saliencies = saliencies.to(device)

    output = model(origs.unsqueeze(0), saliencies.unsqueeze(0))
    output = output.squeeze().cpu().detach()
    output = (output * 255).round().numpy().astype(np.uint8)
    output_saliency = Image.fromarray(output)
    output_saliency.save(saving_path)

def evaluate_model(experiment_folder, orig_folder, m_folder, output_folder):
  model = get_best_model(m_folder)

  os.makedirs(output_folder, exist_ok=True)
  img_list = os.listdir(experiment_folder / Path("layer_1"))
  img_list = [img.replace(".png", "") for img in img_list if "_sal" in img]

  dataset = InferenceSalienciesDataset(
    orig_folder=orig_folder,
    saliencies_folder=experiment_folder,
    img_list=img_list, n_layers=4 if "parasites" in str(output_folder) else 3,
    img_size=(400, 400) if "parasites" in str(output_folder) else (240, 240)
  )

  run_inference(model, dataset, output_folder)

"""
Execution Sample:

python3.11 run_inference.py \
  -or /workdir/miscellaneous/parasites/split1/train_samples/orig/ \
  -ef /workdir/out/parasites/split1/sample_ca_saliencies/ \
  -mf /workdir/out/parasites/split1/merging_model/ \
  -out /workdir/out/parasites/split1/sample_merged_saliencies/
"""

if __name__ == "__main__":
  args = get_inferece_args()
  orig_folder = args.orig_folder
  experiment_folder = args.experiment_folder
  m_folder = args.m_folder
  output_folder = args.output_folder

  print("[INFO] Evaluating model")
  evaluate_model(
    experiment_folder, orig_folder, m_folder, output_folder
  )