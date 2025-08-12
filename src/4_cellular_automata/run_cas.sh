#!/bin/bash
verify_experiment() {
    if [[ "$EXPERIMENT" == "parasites" ]]; then
        echo "4 0"  # 4 layers, IS_BRAIN=0
    elif [[ "$EXPERIMENT" == "brain_tumor" ]]; then
        echo "3 1"  # 3 layers, IS_BRAIN=1
    else
        echo "Error: EXPERIMENT must be either 'parasites' or 'brain_tumor'" >&2
        echo "Current value: '$EXPERIMENT'" >&2
        return 1
    fi
}

INPUT_FOLDER=$1
USE_GPU=$2
EXPERIMENT=$(basename "$INPUT_FOLDER")

# Get both values at once
result=$(verify_experiment)
if [[ $? -eq 0 ]]; then
    read N_LAYERS IS_BRAIN <<< "$result"
else
    exit 1
fi

for SPLIT in split1 split2 split3; do
  # Current split directory (split1, split2, or split3)
  SPLIT_PATH=$INPUT_FOLDER/$SPLIT/
  # Path to sample training images
  ORIG_FOLDER=$SPLIT_PATH/train_samples/orig

  for ((LAYER=1; LAYER<=N_LAYERS; LAYER++)); do
    echo "Processing $SPLIT - Layer $LAYER/$N_LAYERS"

    SALIENCY_FOLDER="/workdir/out/$EXPERIMENT/$SPLIT/sample_saliencies/layer_$LAYER/"
    # Empty, the released implementation does not use Features (We are currently working on that)
    FEATURE_FOLDERS="None"
    OUTPUT_FOLDER="/workdir/out/$EXPERIMENT/$SPLIT/sample_ca_saliencies/layer_$LAYER/"
    echo $SALIENCY_FOLDER

    # Run CPU or GPU Cellular Automata
    if [[ "$USE_GPU" == "0" ]]; then
      iftCA $SALIENCY_FOLDER $ORIG_FOLDER $FEATURE_FOLDERS $OUTPUT_FOLDER $IS_BRAIN
    elif [[ "$USE_GPU" == "1" ]]; then
      iftCAGpu $SALIENCY_FOLDER $ORIG_FOLDER $FEATURE_FOLDERS $OUTPUT_FOLDER $IS_BRAIN
    else
      echo "Error: USE_GPU must be 0 or 1, got: '$USE_GPU'" >&2
      exit 1
    fi
  done
done