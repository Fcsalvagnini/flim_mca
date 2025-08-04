# Feature Learning from Image Markers meets Cellular Automata

This repository contains the source code and experimental materials for the Master's thesis **"Feature Learning from Image Markers meets Cellular Automata"** by Felipe Crispim da Rocha Salvagnini, supervised by Prof. Dr. Alexandre Xavier Falcão at the Institute of Computing, University of Campinas (UNICAMP).

![Multi-Level Cellular Automata for L-Layer FLIM networks.](miscellaneous/images/flim_ca_framework.png)

<p align="center"><i>Figure 1: Multi-Level Cellular Automata for L-Layer FLIM networks.</i></p>

> This document aims to release the method developed during the Master's Degree of Felipe Crispim da Rocha Salvagnini, in a self-contained fashion, so one can validate it using our training images and employ it towards different problems.

> **DISCLAIMER:** Docker container uses CUDA. If you do not have a GPU, just use Cellular Automata and training/inference on CPU (for the merge model)

____

## Abstract

Deep learning approaches typically require extensive annotated datasets and increasingly complex network architectures. This paradigm presents significant challenges in resource-constrained environments, particularly for medical applications in developing countries where data annotation is costly and computational resources are limited. Additionally, many real-world problems, such as detecting Schistosoma mansoni eggs, involve fewer categories than general computer vision tasks, suggesting that simplified approaches may be viable. The Feature Learning from Image Markers (FLIM) methodology enables experts to design convolutional encoders directly from image patches, providing control over encoder complexity. Integrating an adaptive decoder with a FLIM encoder creates networks that eliminate the need for backpropagation and substantially reduce annotation requirements (typically to only 3-4 images). Combining FLIM networks with Cellular Automata (CA) creates a comprehensive pipeline for exploring object detection (or segmentation) on images. The CA works as a post-processing technique; moreover, FLIM facilitates the CA initialization, leveraging user knowledge without requiring per-image user interaction. Therefore, this MSc thesis aims to explore the integration of FLIM networks into the initialization of CA. We evaluate the FLIM-CA framework for salient object detection towards two challenging medical datasets: the detection of S. mansoni eggs in optical microscopy images and brain tumor detection in magnetic resonance imaging data. Our results demonstrate competitive performance compared to convolutional deep learning methods, with improvements of up to 13\% to 20\% on metrics such as F-Score and uWF, using only a fraction of the parameters (thousands vs. millions). Finally, we present a multi-level FLIM-CA to explore the convolutional encoder's hierarchical representation at each level, where intermediary saliency maps initialize corresponding CAs, and the outputs of CAs are merged into a final, improved saliency map. Our work proposes a multi-level FLIM-CA system that builds upon the hierarchical capabilities of FLIM encoders.

____

## Organization

```bash
FLIM_MCA/
├── docker/                      # Docker configuration files for containerized execution
├── miscellaneous/               # Supplementary materials and datasets
│   ├── brain_tumor/             # Brain tumor detection (FLIM files and sample images)
│   ├── images/                  # Documentation images and figures
│   └── parasites/               # Schistosoma mansoni egg detection (FLIM files and sample images)
├── src/                         # Source code organized by pipeline stages
│   ├── 1_flim_design/           # 1: FLIM encoder design
│   ├── 2_feature_extraction/    # 2: Extract features through FLIM Convolutional Encoder
│   ├── 3_decoding/              # 3: Decodes extracted features, characterizing a FLIM network
│   ├── 4_cellular_automata/     # 4: Cellular automata initialization and evolution algorithms
│   └── 5_saliency_fusion/       # 5: Multi-level saliency map fusion and final output generation
├── .gitignore                   # Git ignore configuration
├── LICENSE                      # Project license
└── README.md                    # Project documentation (this file)
```

> **Datasets DISCLAIMER:** This repository releases only the minimal training images necessary to design a FLIM Encoder, run multi-level CAs, and train/test the network that merges multi-level saliencies into a unified version.

> **Parasites Dataset:** Our experiments use a private parasite egg dataset developed in our laboratory, recently made available at [https://github.com/LIDS-Datasets/schistossoma-eggs](https://github.com/LIDS-Datasets/schistossoma-eggs). Training images (with markers) and validation splits are organized in `trainN.csv` files, where N represents the split number (1-3). Test images are listed in `testN.csv` files.

> **BraTS Dataset:** We use the BraTS 2021 dataset for brain tumor experiments. Access to the complete BraTS dataset requires registration and agreement to the challenge terms at the [official BraTS website](https://www.synapse.org/Synapse:syn51156910/wiki/622351). Our repository includes only the minimal training subset used for FLIM encoder design, following the same split structure (`trainN.csv`/`testN.csv`). Following the files, one can process the 3D dataset to generate our 2D dataset, where each file name is structured as: `BraTS2021_00014_a1_s0051.png`, where `sXXXX` indicates the axial slice.

> **Note:** Due to the FLIM methodology's requirement of only 3-4 training images per dataset, the released training subsets are sufficient to reproduce our FLIM encoder design process and validate our multi-level CA approach. After the initial learning, the reader is ready to evaluate the proposed pipeline on different problems.