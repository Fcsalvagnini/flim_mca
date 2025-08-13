#!/bin/bash
INPUT_FOLDER=$1
EXPERIMENT=$(basename "$INPUT_FOLDER")

for SPLIT in split1 split2 split3; do
  ORIG_FOLDER=$INPUT_FOLDER/$SPLIT/train_samples/orig
  LABEL_FOLDER=$INPUT_FOLDER/$SPLIT/train_samples/label
  EXPERIMENT_FOLDER=/workdir/out/$EXPERIMENT/$SPLIT/sample_ca_saliencies
  MODEL_FOLDER=/workdir/out/$EXPERIMENT/$SPLIT/merging_model

  python3.11 train_merge_mode.py -or $ORIG_FOLDER -lf $LABEL_FOLDER \
    -ef $EXPERIMENT_FOLDER -mf $MODEL_FOLDER -e 2000 -vor None -vlf None
done