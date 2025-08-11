from pyflim import layers, arch
import argparse
from pathlib import Path
import os
import pyift.pyift as ift
import torch
import numpy as np
from skimage import io
from tqdm import tqdm

"""
  Sample file: Iterate over the input folder for each split and decode the
sample feature. If you have downloaded the entire dataset or a personal one,
please modify the feature folder names to execute for your dataset.
"""

if __name__ == "__main__":
  parser = argparse.ArgumentParser(
    description='Decode features from multiple split data',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )
    # Add input folder argument
  parser.add_argument(
    'input_folder',
    type=str,
    help='Path to the input folder with split sub-folders'
  )
  args = parser.parse_args()
  device = "cuda" if torch.cuda.is_available() else "cpu"
  input_folder = Path(args.input_folder)
  split_folders = os.listdir(input_folder)
  basefolder = os.path.basename(input_folder)
  output_folder = Path(f"/workdir/out/{basefolder}")

  for split in split_folders:
    features_path = input_folder / split / "sample_features"
    layer_folders = os.listdir(features_path)
    for layer in layer_folders:
      layer_path = features_path / layer
      images = os.listdir(layer_path)

      sample_image = ift.ReadMImage(str(layer_path / images[0])).AsNumPy()
      ch = sample_image.shape[-1]
      decoder = layers.FLIMAdaptiveDecoderLayer(
        ch,
        adaptation_function="robust_weights",
        decoder_type="vanilla_adaptive_decoder",
        filter_by_size=False,
        device=device
      )
      saving_path = output_folder / split / "sample_saliencies" / layer
      if not os.path.exists(saving_path):
        os.makedirs(saving_path)
      # Read first instantiate decoder
      for img in tqdm(images):
        img_path = layer_path / img
        mimg_array = ift.ReadMImage(str(img_path)).AsNumPy()
        mimg_tensor = torch.from_numpy(mimg_array)
        mimg_tensor = mimg_tensor.clone().to(device).permute(0, 3, 1, 2)
        saliency = decoder(mimg_tensor)
        saliency = saliency.squeeze().detach().numpy().astype(np.uint8)

        img_saving_path = saving_path / img.replace("mimg", "png")
        io.imsave(img_saving_path, saliency, check_contrast=False)

      del decoder