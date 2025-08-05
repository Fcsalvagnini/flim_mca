#!/bin/bash
FLIM_FILES=$1
EXPERIMENT=$(basename "$FLIM_FILES")

ARCHITECTURE_FILE=$FLIM_FILES/arch2D.json
for SPLIT in split1 split2 split3; do
  # Current split directory (split1, split2, or split3)
  SPLIT_PATH=$FLIM_FILES/$SPLIT/
  # Path to sample training images
  ORIG_FOLDER=$SPLIT_PATH/train_samples/orig
  # Path for markers. Each training image has an associate marker file,
  # which specifies the locations marked by the user to guide the FLIM Encoder
  # learning phase!
  MARKERS_FOLDER=$SPLIT_PATH/markers
  OUT_FOLDER=/workdir/out/$EXPERIMENT/$SPLIT/flim_model

  iftFLIM-LearnModel $ARCHITECTURE_FILE $ORIG_FOLDER $MARKERS_FOLDER $OUT_FOLDER
done