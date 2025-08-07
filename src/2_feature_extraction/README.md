# FLIM feature extraction: Extract features using the learned FLIM Convolutional Encoder

With an FLIM encoder in place, we are now ready to extract feature maps. In this example, we will extract features for the sample training images only. However, if you have the complete datasets (train/validation + test), you can proceed and run for the whole subsets without significant changes. Follow the two steps below:

1. Compile the `iftFLIM-ExtractFeaturesFromLayer.c` code `./compile.sh 1` for GPU support or `./compile.sh 0` for CPU;
2. Extract the features for each split, using the sample training images. Run the `extract_features.sh` script:

```bash
# For parasite samples
./extract_features.sh /workdir/miscellaneous/parasites/
# For brain samples
./extract_features.sh /workdir/miscellaneous/brain_tumor/
```

The script calls the `iftFLIM-ExtractFeaturesFromLayer` executable, with the following arguments:

1. The FLIM encoder architecture (specified through a json file);
2. The folder with the input images (e.g., colored or gray-scale images of our problem) or the feature maps from the previous layers (.mimg);
3. The list of images or features for feature extraction. Here, we are passing `sample.csv` or `samplem.csv`, where the former specifies sample images, and the latter specifies sample features (`.mimg`). If you have downloaded the whole dataset for parasites, or prepared the BraTS 2D dataset, you can pass the split files for complete dataset execution (e.g., `train1.csv`, `test1.csv` or `train1m.csv`, `test1m.csv`, for split 1, for instance);
4. Input folder with FLIM Encoder learned parameters
5. Index of layer to load weights
6. Output folder to save extracted features (`.mimg` format)
7. GPU index to use (if using GPU)

____

Once the FLIM Encoder weights are loaded, the operations are straightforward and well-known, following the steps below:

1. Input images, or features, are normalized using the learned MBIN parameters
2. Inputs are convolved by the corresponding encoder block, with subsequent operations such as the ReLU activation function and pooling operations.