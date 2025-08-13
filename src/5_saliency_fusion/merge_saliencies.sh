#!/bin/bash
INPUT_FOLDER=$1
EXPERIMENT=$(basename "$INPUT_FOLDER")

for SPLIT in split1 split2 split3; do
  ORIG_FOLDER=$INPUT_FOLDER/$SPLIT/train_samples/orig
  EXPERIMENT_FOLDER=/workdir/out/$EXPERIMENT/$SPLIT/sample_ca_saliencies
  MODEL_FOLDER=/workdir/out/$EXPERIMENT/$SPLIT/merging_model
  OUTPUT_FOLDER=/workdir/out/$EXPERIMENT/$SPLIT/sample_merged_saliencies

  python3.11 run_inference.py -or $ORIG_FOLDER -ef $EXPERIMENT_FOLDER \
                              -mf $MODEL_FOLDER -out $OUTPUT_FOLDER
done