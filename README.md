# Lab 2 — Computer Vision Architectures

> **Deep Learning | Faculté des Sciences et Techniques de Tanger (FSTT)**
> Département de Génie Informatique — Cycle Ingénieur LSI
> Module encadré par **Pr. Lotfi ELAACHAK** | Réalisé par **Mohammed Salhi**

This project implements, trains, and benchmarks five major computer vision architectures on MNIST — from a hand-built CNN to a Vision Transformer assembled from scratch — in pure **PyTorch**.

---

## Contents

1. [Overview](#overview)
2. [Project Structure](#project-structure)
3. [Quick Start](#quick-start)
4. [Part 1 — Convolutional Architectures](#part-1--convolutional-architectures)
   - [Custom CNN Classifier](#custom-cnn-classifier)
   - [Faster R-CNN Object Detector](#faster-r-cnn-object-detector)
   - [CNN vs. Faster R-CNN — Comparison](#cnn-vs-faster-r-cnn--comparison)
   - [Transfer Learning — AlexNet & VGG16](#transfer-learning--alexnet--vgg16)
5. [Part 2 — Vision Transformer from Scratch](#part-2--vision-transformer-from-scratch)
   - [Architecture](#architecture)
   - [Implementation Walkthrough](#implementation-walkthrough)
   - [Training Results](#training-results)
   - [Attention Map Visualization](#attention-map-visualization)
   - [CNNs vs. ViT — Discussion](#cnns-vs-vit--discussion)
6. [Full Benchmark](#full-benchmark)
7. [Key Takeaways](#key-takeaways)
8. [References](#references)

---

## Overview

Five architectures are studied, each representing a distinct design philosophy:

| Architecture | Paradigm | Test Accuracy | Parameters |
| :--- | :--- | :---: | :---: |
| Custom CNN | Spatial convolution | **97.90%** | 620 K |
| Faster R-CNN | Two-stage detection | 3.75% * | 18.9 M |
| AlexNet (fine-tuned) | Transfer learning | 91.00% | 54.6 M |
| VGG16 (fine-tuned) | Transfer learning | 92.10% | 119.6 M |
| ViT (from scratch) | Self-attention | 80.97% | 83.5 K |

> \* Faster R-CNN is a **detection** model, not a classifier. Its primary metric is bounding box IoU (0.52 – 0.62), not top-1 classification accuracy.

All models are evaluated under the same conditions: identical test split, same hardware, and a shared set of metrics — accuracy, macro F1, cross-entropy loss, training time, inference latency, and parameter count.

---

## Project Structure

```
deep_learning_atelier2/
│
├── src/
│   ├── data_utils.py              # MNIST loaders, augmentation, detection dataset
│   ├── cnn_classifier.py          # Custom 3-block CNN with BatchNorm and Dropout
│   ├── faster_rcnn_detector.py    # Faster R-CNN adapted for MNIST digit detection
│   ├── transfer_learning.py       # AlexNet and VGG16 fine-tuning pipeline
│   ├── vit_scratch.py             # Vision Transformer built component by component
│   └── benchmark.py               # Runs all experiments and generates all figures
│
├── notebooks/
│   └── Vision_Architectures_Lab2.ipynb   # Self-contained notebook version
│
├── scripts/
│   └── generate_notebook.py       # Regenerates the notebook from source
│
├── figures/                       # All output plots and confusion matrices
│   ├── cnn_training_curves.png
│   ├── cnn_confusion_matrix.png
│   ├── faster_rcnn_loss_curve.png
│   ├── faster_rcnn_predictions.png
│   ├── transfer_learning_comparison.png
│   ├── transfer_confusion_matrices.png
│   ├── vit_training_curves.png
│   ├── vit_confusion_matrix.png
│   ├── vit_attention_maps.png
│   ├── model_metrics_comparison_barcharts.png
│   ├── architecture_tradeoff_scatter.png
│   └── benchmark_comparison.csv
│
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
# Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install all dependencies
pip install -r requirements.txt
```

Run any individual module:

```bash
python src/cnn_classifier.py          # Part 1.1 — Custom CNN
python src/faster_rcnn_detector.py    # Part 1.2 — Faster R-CNN
python src/transfer_learning.py       # Part 1.4 — AlexNet & VGG16
python src/vit_scratch.py             # Part 2   — Vision Transformer
python src/benchmark.py               # Full benchmark suite + all figures
```

Regenerate the notebook:

```bash
python scripts/generate_notebook.py
# Output: notebooks/Vision_Architectures_Lab2.ipynb
```

---

## Part 1 — Convolutional Architectures

### Custom CNN Classifier

A compact, fully regularized convolutional network trained end-to-end on MNIST from random initialization.

**Architecture — `MNISTConvNet`**

```
Input  (1 x 28 x 28)
  |
  +-- Conv2D(1->32, k=3, p=1) -- BatchNorm2d -- ReLU -- MaxPool(2)   =>  32 x 14 x 14
  |
  +-- Conv2D(32->64, k=3, p=1) -- BatchNorm2d -- ReLU -- MaxPool(2)  =>  64 x 7 x 7
  |
  +-- Conv2D(64->128, k=3, p=1) -- BatchNorm2d -- ReLU -- AdaptiveAvgPool(3x3)  =>  128 x 3 x 3
  |
  +-- Flatten -- Linear(1152->128) -- ReLU -- Dropout(0.4) -- Linear(128->10)
```

**Regularization strategy**

- `BatchNorm2d` after every convolution — stabilizes gradients, speeds convergence
- `Dropout(p=0.4)` in the classifier head — prevents co-adaptation of neurons
- `AdamW` optimizer with weight decay `1e-4` — penalizes large weights
- `CosineAnnealingLR` scheduler — smoothly decays learning rate from `1e-3` to `1e-5`

**Results after 5 epochs**

| Metric | Train | Test |
| :--- | :---: | :---: |
| Cross-Entropy Loss | 0.0452 | 0.0584 |
| Accuracy | 98.63% | **97.90%** |
| Macro F1 | 0.9862 | **0.9791** |
| Inference Latency | — | 1.15 ms / batch |

![CNN Training Curves](figures/cnn_training_curves.png)
*Loss and accuracy curves — smooth convergence with no signs of overfitting.*

![CNN Confusion Matrix](figures/cnn_confusion_matrix.png)
*Normalized confusion matrix — above 97% accuracy on every digit class.*

---

### Faster R-CNN Object Detector

Unlike classification, object detection requires simultaneously predicting **where** and **what** each digit is. The model outputs a set of bounding boxes each paired with a class label and confidence score.

**Model pipeline**

```
Input (3 x 56 x 56)
  |
  +-- MobileNetV3 Backbone + Feature Pyramid Network (FPN)
        |
        +-----------------------------+
        |                             |
  Region Proposal Network       Feature Maps
        |                             |
  Candidate RoIs  -----> RoI Align (7x7 pooling)
                                      |
                            Fast R-CNN Predictor Head
                                |              |
                      Class scores (11)   Box deltas (4 x 11)
```

**Training setup**

- Optimizer: SGD with momentum 0.9, weight decay 5e-4
- Multi-task loss: RPN classification + RPN box regression + RoI classification + RoI box regression
- Image size fixed at 56x56 to avoid default upsampling overhead

**Key metrics**

| Metric | Value |
| :--- | :---: |
| Mean Bounding Box IoU | 0.52 – 0.62 |
| Inference Latency | 27.52 ms / batch |
| Parameter Count | 18,917,458 |

![Faster R-CNN Loss Curve](figures/faster_rcnn_loss_curve.png)
*Multi-task training loss — all four loss components decrease consistently.*

![Faster R-CNN Predictions](figures/faster_rcnn_predictions.png)
*Ground truth boxes (green) vs. predicted boxes (red) with class labels and confidence scores.*

---

### CNN vs. Faster R-CNN — Comparison

These two architectures solve fundamentally different problems and should not be compared on accuracy alone.

| Dimension | Custom CNN | Faster R-CNN |
| :--- | :--- | :--- |
| Task | Single-label classification | Localization + classification |
| Output | 10 class logits | Variable-length list of (box, class, score) |
| Parameters | 620 K | 18.9 M (30x larger) |
| Inference latency | 1.15 ms | 27.52 ms (24x slower) |
| Top-1 accuracy | **97.90%** | 3.75% (strict match) |
| Spatial output | None | Bounding boxes, IoU ~0.52–0.62 |
| Best suited for | Isolated digit recognition | Multi-object scenes, document parsing |

> The CNN wins on every efficiency metric when the task is classification.
> Faster R-CNN is the only option when spatial localization is required.

---

### Transfer Learning — AlexNet & VGG16

Pre-trained ImageNet backbones are adapted to MNIST via **linear probing**: the feature extractor is frozen, and only the final classification head is retrained.

**Adaptation steps**

1. Grayscale images (1 x 28 x 28) are expanded to 3 channels and resized to 64 x 64
2. All backbone parameters are frozen (`requires_grad = False`)
3. The final fully-connected layer is replaced:
   - AlexNet: `classifier[6] = nn.Linear(4096, 10)`
   - VGG16: `classifier[6] = nn.Linear(4096, 10)`

**Results after 2 epochs**

| Model | Accuracy | Macro F1 | Test Loss | Latency | Parameters |
| :--- | :---: | :---: | :---: | :---: | :---: |
| AlexNet | 91.00% | 0.9079 | 0.2706 | 18.69 ms | 54.6 M |
| VGG16 | 92.10% | 0.9194 | 0.2556 | 62.59 ms | 119.6 M |
| Custom CNN | **97.90%** | **0.9791** | **0.0584** | **1.15 ms** | **0.62 M** |

![Transfer Learning Comparison](figures/transfer_learning_comparison.png)
*Validation loss and accuracy for AlexNet and VGG16 during fine-tuning.*

![Transfer Learning Confusion Matrices](figures/transfer_confusion_matrices.png)
*Side-by-side confusion matrices — AlexNet (left) and VGG16 (right).*

> Both transfer models adapt quickly, but the Custom CNN outperforms them with 192x fewer parameters. Large ImageNet backbones carry excessive capacity for 28x28 grayscale strokes, and the domain gap between natural photographs and handwritten digits limits how much pretrained knowledge transfers.

---

## Part 2 — Vision Transformer from Scratch

### Architecture

The Vision Transformer (ViT) abandons local convolutions entirely. Instead, it divides the image into a fixed grid of patches and applies global self-attention across all patches simultaneously from the very first layer.

```
Input (1 x 28 x 28)
  |
  +-- Patchify (7x7 grid) --> 49 patches, each of size 4x4x1 = 16 dimensions
  |
  +-- Linear Projection (16 -> 32) + Prepend [CLS] token + Sinusoidal Positional Encoding
  |                                             (sequence shape: B x 50 x 32)
  |
  +-- [ ViT Encoder Block x 2 ]
  |     |
  |     +-- LayerNorm --> Multi-Head Self-Attention (2 heads) --> Residual Add
  |     |
  |     +-- LayerNorm --> MLP (32 -> 64 -> 32)               --> Residual Add
  |
  +-- Extract [CLS] token --> LayerNorm --> Linear (32 -> 10) --> Logits
```

---

### Implementation Walkthrough

#### Step 1 — Patchification

The image is divided into non-overlapping patches and each patch is flattened into a 1D vector.

For H = W = 28, patch size P = 4:

- Number of patches: N = (28 / 4) x (28 / 4) = 49
- Patch dimension: D = 4 x 4 x 1 = 16

```python
def patchify(images, n_patches=7):
    B, C, H, W = images.shape
    patch_size = H // n_patches
    patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
    return patches.view(B, n_patches * n_patches, -1)
```

#### Step 2 — Positional Encoding

Self-attention has no built-in sense of order. Sinusoidal embeddings inject position information into the sequence:

$$PE_{(pos,\ 2i)} = \sin\!\left(\frac{pos}{10000^{2i/d}}\right), \qquad PE_{(pos,\ 2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d}}\right)$$

#### Step 3 — Multi-Head Self-Attention

Each token produces Query, Key, and Value vectors. Attention scores are computed as scaled dot products, then used to aggregate values:

$$\text{Attention}(Q, K, V) = \text{Softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

```python
class MyMSA(nn.Module):
    def __init__(self, d, n_heads=2):
        super().__init__()
        self.n_heads = n_heads
        self.d_head  = d // n_heads
        self.q_mappings = nn.Linear(d, d)
        self.k_mappings = nn.Linear(d, d)
        self.v_mappings = nn.Linear(d, d)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        B, N, d = x.shape
        q = self.q_mappings(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_mappings(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_mappings(x).view(B, N, self.n_heads, self.d_head).transpose(1, 2)
        scores  = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        weights = self.softmax(scores)
        out     = torch.matmul(weights, v).transpose(1, 2).contiguous().view(B, N, d)
        return out, weights
```

#### Step 4 — Encoder Block

Pre-LayerNorm residual structure following the original ViT paper:

$$z'_l = \text{MSA}(\text{LN}(z_{l-1})) + z_{l-1}$$

$$z_l = \text{MLP}(\text{LN}(z'_l)) + z'_l$$

---

### Training Results

Trained for 5 epochs with Adam at learning rate 5e-3:

| Metric | Value |
| :--- | :---: |
| Test Accuracy | **80.97%** |
| Macro F1 | **0.8033** |
| Test Loss | **0.5738** |
| Parameters | **83,466** (83.5K) |
| Inference Latency | **1.61 ms / batch** |

![ViT Training Curves](figures/vit_training_curves.png)
*Steady loss reduction and accuracy gain over 5 epochs — no convolutions used.*

![ViT Confusion Matrix](figures/vit_confusion_matrix.png)
*Confusion matrix — solid per-class performance across all 10 digit categories.*

---

### Attention Map Visualization

The `[CLS]` token aggregates information from all 49 patches to form the final classification decision. By extracting its row from the attention matrix and reshaping it to a 7x7 grid, we get a spatial heatmap showing which regions the model attended to.

$$M = \text{Reshape}_{7 \times 7}\!\left(A\bigl[\text{head},\; 0,\; 1{:}50\bigr]\right)$$

![ViT Attention Maps](figures/vit_attention_maps.png)
*Top row: original digit. Bottom row: attention heatmap. High-intensity patches align with active digit strokes.*

Key observations:

- **Stroke saliency**: For digits 0, 3, and 8, attention peaks at loop junctions and curvature boundaries
- **Background suppression**: Near-zero attention on blank patches — the model learned to ignore empty regions without any explicit supervision

---

### CNNs vs. ViT — Discussion

| Property | CNN | Vision Transformer |
| :--- | :--- | :--- |
| Core operation | Local convolution (K x K kernel) | Global scaled dot-product attention |
| Inductive bias | Strong: locality + translation equivariance | Minimal: only 1D positional encoding |
| Receptive field | Grows with depth | Full image from layer 1 |
| Data efficiency | High — strong priors need less data | Low — must learn geometry from data |
| Computational cost | Linear in image size | Quadratic in patch count |
| Interpretability | Requires post-hoc methods (Grad-CAM) | Built-in via attention weights |
| MNIST result | 97.90% (5 epochs) | 80.97% (5 epochs) |

> The accuracy gap narrows significantly with more epochs, larger datasets, or pretraining. ViTs consistently outperform CNNs at scale — the tradeoff is data and compute efficiency at the smaller end.

---

## Full Benchmark

All five models evaluated on the same MNIST test partition under identical hardware conditions:

| Model | Accuracy | F1 | Loss | Train Time | Latency | Params |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Custom CNN | **97.90%** | **0.9791** | **0.0584** | 68.3 s | 1.15 ms | 620 K |
| Faster R-CNN | 3.75% | 0.0072 | 3.9327 | 29.4 s | 27.52 ms | 18.9 M |
| AlexNet FT | 91.00% | 0.9079 | 0.2706 | 108.7 s | 18.69 ms | 54.6 M |
| VGG16 FT | 92.10% | 0.9194 | 0.2556 | 220.8 s | 62.59 ms | 119.6 M |
| ViT Scratch | 80.97% | 0.8033 | 0.5738 | 33.5 s | 1.61 ms | 83.5 K |

![Model Metrics Comparison](figures/model_metrics_comparison_barcharts.png)
*Bar charts across all six evaluation metrics.*

![Architecture Tradeoff](figures/architecture_tradeoff_scatter.png)
*Latency vs. accuracy scatter — bubble size proportional to parameter count.*

---

## Key Takeaways

**Inductive biases matter at small scale.**
The Custom CNN reaches 97.90% because convolutional priors (locality, translation equivariance) are perfectly suited to small, centered digit images. The ViT achieves only 80.97% from scratch — it needs more data or longer training to compensate for the absence of these priors.

**Classification and detection are different problems.**
Comparing Faster R-CNN accuracy to CNN accuracy is misleading. Faster R-CNN's value is spatial — it outputs bounding boxes that classification models cannot produce. For pure classification, CNNs are 30x smaller and 24x faster.

**Transfer learning has diminishing returns in narrow domains.**
AlexNet and VGG16 reach ~91–92% with just 2 epochs, which is impressive. But with 54M–120M parameters vs. 620K in the Custom CNN, the computational overhead is disproportionate for a 28x28 grayscale task. Domain gap (natural images vs. digit strokes) limits what transfers.

**Attention provides free interpretability.**
No post-hoc analysis is needed with ViT. The attention weights directly expose which image regions drove the classification — a built-in advantage for explainability-critical applications.

---

## References

1. Dosovitskiy, A., et al. (2020). *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.* ICLR 2021. arXiv:2010.11929.
2. Ren, S., He, K., Girshick, R., & Sun, J. (2015). *Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks.* NeurIPS 2015. arXiv:1506.01497.
3. Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). *ImageNet Classification with Deep Convolutional Neural Networks.* NeurIPS 2012.
4. Simonyan, K., & Zisserman, A. (2014). *Very Deep Convolutional Networks for Large-Scale Image Recognition.* ICLR 2015. arXiv:1409.1556.
5. Pulfer, B. (2022). *Vision Transformers from Scratch (PyTorch): A step-by-step guide.* Towards Data Science.
6. Faculté des Sciences et Techniques de Tanger — Module Deep Learning, Pr. Lotfi ELAACHAK.
