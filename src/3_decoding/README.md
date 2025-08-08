# FLIM feature decodification: Adaptively decode a saliency map from FLIM's extracted features

In this step, we will be using two Python modules developed by our laboratory colleagues:

- **[pyift](https://github.com/LIDS-UNICAMP/pyift)**: We will use it to read our feature maps (.mimg)
- **[pyflim](https://github.com/LIDS-UNICAMP/flim-python-demo)**: We will use them to decode each layer feature map into intermediary saliency maps, employed for CA Initialization.

First, we need to solve all dependencies and install the required modules.

Let us first install those modules! We have released installation files at `lib_flim_mca`. Follow the instructions below:

```bash
# Navigate into the lib_flim_mca folder
cd /workdir/lib_flim_mca/lib_pyflim

# First install pyift
cd lib_pyift/
./install_pyift.sh
cd ..

# Secondly, install pyflim
cd lib_pyflim/
python3.11 -m pip install pyflim-0.1-py3-none-any.whl

# Also install the tqdm package
python3.11 -m pip install tqdm
```

Then, we are ready to decode our extracted features. Code snippets below exemplify decoding our sample features:

```bash
# Decode parasites features
python3.11 decode_features.py /workdir/out/parasites/

# Decode brain tumor features
python3.11 decode_features.py /workdir/out/brain_tumor/
```

The Python script will decode sample saliencies into the folders: `out/parasites/splitN/sample_saliencies/layer_L` and `out/brain_tumor/splitN/sample_saliencies/layer_L` for parasite eggs and brain tumors, respectively. 

It is important to note that extracted saliencies are smaller than the input image, which happens because of the stride operations.

Summarizing, the Python script does the following:

1. Iterate over all features (for each split, and layer);
2. Given the input features and their number of channels, it adaptively decodes the features by weighting each feature map channel;
3. Decoded saliency is then saved in output folders.

The adaptive decoding approach employed in this work was developed and proposed by our laboratory colleagues, building on the foundational work by [João et al. (2023)](https://arxiv.org/abs/2306.14840v1), who introduced adaptive decoders as a key innovation in FLIM CNN architectures. This work adopts the modified approach from [João et al. (2024)](https://pdfs.semanticscholar.org/3088/68875a31ea8fb543e03985708e50bccd8cb6.pdf), which effectively addresses the challenge of dynamic channel polarity in FLIM feature maps.

Optimal saliency detection emerges when decoders assign positive weights to foreground-selective channels and negative weights to background-selective channels, effectively suppressing false positive responses. However, channel polarity varies dynamically with input content, demanding adaptive decoding (i.e., weighting) strategies.

The adaptive decoder computes saliency maps **S<sub>l</sub>** according to the following equation:

**S<sub>l</sub>(p) = φ(⟨J<sub>l</sub>(p), w⟩)**

Where:
- **w** = (w¹, w², ..., w^(n×M×|C_I|)) represents channel weights
- **J<sub>l</sub>(p)** denotes the feature vector at spatial position p for layer l  
- **φ** implements the chosen activation function (typically ReLU)

Channel weights **w<sup>i</sup> ∈ {-1, 0, 1}** are determined adaptively through statistical analysis:

$$w^i = \begin{cases}
 +1, & \text{if } \mu_{J^i} \leq \tau - \sigma^2_{J^i} \text{ and } a^i < A_1 \\
 -1, & \text{if } \mu_{J^i} \geq \tau + \sigma^2_{J^i} \text{ and } a^i > A_2 \\
    0, & \text{otherwise}
\end{cases}$$

Where:
- **μ<sub>J<sup>i</sup></sub>**: Channel mean activations
- **σ²<sub>J<sup>i</sup></sub>**: Channel variance
- **τ**: Adaptive threshold
- **a<sup>i</sup>**: Activation ratio
- **A₁, A₂**: Area thresholds (empirically set to A₁ = 0.1 and A₂ = 0.2)


The Python script automatically applies this adaptive decoding strategy to each feature map channel, analyzing the statistical properties of activations to determine optimal weighting schemes, ensuring robust saliency map generation across diverse input content while maintaining computational efficiency.

For further details, please review our background and related work section on Feature Learning from Image Markers, where additional information is provided.