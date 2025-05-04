
# 🚀 WGAN.ipynb – Wasserstein GAN on CelebA Dataset

This notebook implements a **Wasserstein Generative Adversarial Network (WGAN)** using PyTorch to generate realistic images based on the CelebA dataset. Designed with clarity and modularity in mind, it is suitable for both **research** and **learning** purposes.

---

## 📌 Introduction

Wasserstein GANs improve traditional GANs by replacing the Jensen-Shannon divergence with the **Wasserstein (Earth Mover's) distance**, offering:

* Improved training stability ⚖️
* Better gradient flow for the generator 🎯
* More realistic outputs over time 🖼️

This notebook trains a WGAN on the **CelebA dataset**, focusing on facial image generation and evaluating performance using **Inception Score**, **Frechet Inception Distance (FID)**, and a downstream **classifier**.

---

## ⚙️ Environment Setup

| Component   | Details                            |
| ----------- | ---------------------------------- |
| Language    | Python 3.7+                        |
| Libraries   | PyTorch, Torchvision, Numpy, etc.  |
| Accelerator | ✅ CUDA (GPU support auto-detected) |

Ensure a CUDA-compatible GPU is available for optimal training performance. Training time per epoch: ⏱️ \~9.5 minutes on average with GPU.

---

## 📦 Dataset Acquisition & Preparation

* Dataset: [CelebA (Kaggle)](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)
* Downloaded via `opendatasets` using Kaggle API credentials.
* Structure:

  * `img_align_celeba/` - Image files
  * `list_attr_celeba.csv` - Attribute labels
  * `list_eval_partition.csv` - Train/Val/Test splits

---

## 🧹 Data Loading & Preprocessing

| Task          | Details                                            |
| ------------- | -------------------------------------------------- |
| Image Resize  | 64x64 pixels                                       |
| Normalization | Pixel values scaled to \[-1, 1]                    |
| Label         | Binary "Young" attribute selected for conditioning |
| Datasets      | `train_dataset`, `val_dataset`, `test_dataset`     |

A custom PyTorch Dataset class `CelebADataset` is used for loading and preprocessing.

---

## 🧠 Model Architecture (WGAN)

| Component | Description                                                               |
| --------- | ------------------------------------------------------------------------- |
| Generator | Converts random noise to 64x64 images using ConvTranspose layers.         |
| Critic    | Evaluates realness of images using CNNs (no sigmoid, outputs raw scores). |
| Loss      | Wasserstein loss; critic trained more frequently than generator.          |
| Lipschitz | Enforced via weight clipping or gradient penalty.                         |

---

## 🏋️ Training Procedure

**Epochs:** 3
**Critic Iterations per Generator Update:** 5
**Batch Size:** 64
**Learning Rate:** 0.00005
**Optimizer:** RMSProp

### ⏱️ Training Summary

| Epoch | D\_Loss | G\_Loss | Duration    |
| ----- | ------- | ------- | ----------- |
| 0     | -9.7381 | 40.8979 | \~9 min 32s |
| 1     | -7.0782 | 32.9996 | \~9 min 28s |
| 2     | -6.6255 | 30.1054 | \~9 min 27s |

---

## 📊 Evaluation & Visualization

Generated samples are visualized after each epoch for qualitative assessment. Quantitative metrics:

### 📈 Inception Score & FID

| Metric                  | Value      |
| ----------------------- | ---------- |
| Inception Score (Mean)  | **2.87**   |
| Inception Score (Std)   | ±0.29      |
| Frechet Inception Dist. | **175.54** |

---

## 🧪 Classifier Evaluation (on Real + Synthetic Data)

A classifier was trained on real + WGAN-generated images to validate sample realism.

### 🧮 Training Summary

| Epoch | Loss   | Accuracy   |
| ----- | ------ | ---------- |
| 1     | 0.4317 | 81.88%     |
| 2     | 0.3754 | 84.22%     |
| 3     | 0.3481 | 85.40%     |
| 4     | 0.3230 | 86.61%     |
| 5     | 0.3053 | **87.39%** |

### 📊 Final Evaluation on Test Set

| Metric    | Value      |
| --------- | ---------- |
| Accuracy  | **82.51%** |
| Precision | 83.65%     |
| Recall    | 95.58%     |
| F1-score  | **89.22%** |

---

## 🧰 Customization & Extensions

| Feature             | Description                                               |
| ------------------- | --------------------------------------------------------- |
| 🔄 Attribute Swap   | Use different CelebA attributes (e.g., "Male", "Smiling") |
| 🖼️ Higher Res      | Change image size to 128x128 or 256x256                   |
| 🧪 Conditional GANs | Integrate class labels into generator and discriminator   |
| 🧠 Model Variants   | Upgrade to WGAN-GP, DCGAN, or BigGAN                      |

---

## 📦 Requirements

```bash
Python >= 3.7
torch >= 1.10
torchvision
pandas
numpy
matplotlib
opendatasets
kaggle
scikit-learn
PIL
```

---

## ▶️ Usage Instructions

1. **Download Notebook**
   Place `WGAN.ipynb` in your working directory.

2. **Install Dependencies**
   Run the setup cell in the notebook.

3. **Download CelebA**
   Use your Kaggle API credentials when prompted.

4. **Run Notebook**
   Execute all cells from top to bottom.

5. **Add WGAN Model & Training Code**
   Insert your architecture if not already defined.

6. **Visualize Outputs**
   Inspect generated images and monitor training losses.

---

## 🙏 Acknowledgements

* Dataset: **CelebA** by Chinese University of Hong Kong
* Libraries: PyTorch, Torchvision, OpenDatasets, Scikit-learn, Matplotlib

---

## 💬 Notes

* Trained on GPU for best performance ⚡
* Ensure \~1.3GB free disk space for dataset 📦
* Modular structure allows quick adaptation to new datasets 📁


