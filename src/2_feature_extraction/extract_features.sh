#!/bin/bash
verify_experiment() {
    if [[ "$EXPERIMENT" == "parasites" ]]; then
        echo "4"
    elif [[ "$EXPERIMENT" == "brain_tumor" ]]; then
        echo "3"
    else
        echo "Error: EXPERIMENT must be either 'parasites' or 'brain_tumor'" >&2
        echo "Current value: '$EXPERIMENT'" >&2
        return 1
    fi
}

INPUT_FOLDER=$1
EXPERIMENT=$(basename "$INPUT_FOLDER")
ARCHITECTURE_FILE=$INPUT_FOLDER/arch2D.json
N_LAYERS=$(verify_experiment)

for SPLIT in split1 split2 split3; do
  # Current split directory (split1, split2, or split3)
  SPLIT_PATH=$INPUT_FOLDER/$SPLIT/
  # Path to sample training images
  ORIG_FOLDER=$SPLIT_PATH/train_samples/orig
  # Path to parameters of the FLIM Encoder
  FLIM_MODEL_FOLDER=/workdir/out/$EXPERIMENT/$SPLIT/flim_model

  for ((LAYER=1; LAYER<=N_LAYERS; LAYER++)); do
    echo "Processing $SPLIT - Layer $LAYER/$N_LAYERS"
        
    # Set input folder based on layer and also list of files to extract features
    if [[ $LAYER -eq 1 ]]; then
        FILE_LIST=$SPLIT_PATH/sample.csv
    else
        PREV_LAYER=$((LAYER-1))
        ORIG_FOLDER=/workdir/out/$EXPERIMENT/$SPLIT/sample_features/layer_$PREV_LAYER/
        FILE_LIST=$SPLIT_PATH/samplem.csv
    fi
    OUT_FOLDER=/workdir/out/$EXPERIMENT/$SPLIT/sample_features/layer_$LAYER/

    # Run FLIM feature extraction command
    iftFLIM-ExtractFeaturesFromLayer $ARCHITECTURE_FILE $ORIG_FOLDER \
      $FILE_LIST $FLIM_MODEL_FOLDER $LAYER $OUT_FOLDER 0
  done
done