from pathlib import Path
import argparse

def none_or_path(string):
  if string == 'None':
    return None
  return Path(string)

def get_train_args():
  parser = argparse.ArgumentParser(
      description="Train simple merging models for CA Saliencies",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )

  parser.add_argument(
    '-or', '--orig_folder',
    type=Path,
    default=None,
    required=True,
    help='Path to orig images'
  )

  parser.add_argument(
    '-lf', '--label_folder',
    type=Path,
    default=None,
    required=True,
    help='Path to label images'
  )

  parser.add_argument(
    '-ef', '--experiment_folder',
    type=Path,
    default=None,
    required=True,
    help='Path to experiments folder (with splits)'
  )

  parser.add_argument(
    '-mf', '--m_folder',
    type=Path,
    default=None,
    required=True,
    help='Path to save models'
  )

  parser.add_argument(
    '-e', '--epochs',
    type=int,
    default=None,
    required=True,
    help='Epochs to train the model'
  )

  parser.add_argument(
    '-vor', '--val_orig_folder',
    type=none_or_path,
    default=None,
    required=False,
    help='Path to validation orig images (optional)'
  )

  parser.add_argument(
    '-vlf', '--val_label_folder',
    type=none_or_path,
    default=None,
    required=False,
    help='Path to validation label images (optional)'
  )

  return parser.parse_args()

def get_inferece_args():
  parser = argparse.ArgumentParser(
      description="Evaluates simple merging models for CA Saliencies",
      formatter_class=argparse.ArgumentDefaultsHelpFormatter
  )

  parser.add_argument(
    '-or', '--orig_folder',
    type=Path,
    default=None,
    required=True,
    help='Path to orig images'
  )

  parser.add_argument(
    '-ef', '--experiment_folder',
    type=Path,
    default=None,
    required=True,
    help='Path to experiments folder (with splits)'
  )

  parser.add_argument(
    '-mf', '--m_folder',
    type=Path,
    default=None,
    required=True,
    help='Path to save models'
  )

  parser.add_argument(
    '-out', '--output_folder',
    type=Path,
    default=None,
    required=True,
    help='Folder to save merged saliencies'
  )

  return parser.parse_args()