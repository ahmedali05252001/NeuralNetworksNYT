# 👗 Fashion MNIST — Neural Network Classifier

A deep learning project that trains and compares two neural network architectures on the **Fashion MNIST** dataset, with full evaluation including loss curves, accuracy plots, confusion matrix, and AUC score.

---

## 📌 Project Overview

This project fulfills the following requirements:

| Requirement | Status |
|---|---|
| Load & normalize a dataset | ✅ Fashion MNIST (60,000 train / 10,000 test images) |
| Visualize training errors (Loss & Accuracy) | ✅ Validation loss & accuracy comparison plots |
| Compute AUC score | ✅ One-vs-Rest multiclass AUC |
| Confusion Matrix | ✅ Heatmap via Seaborn |
| Compare Learning Rate, Batch Size, Epochs | ✅ Two models with different hyperparameters |

---

## 📂 Dataset

**Fashion MNIST** — loaded directly via `keras.datasets.fashion_mnist`

- 70,000 grayscale images (28×28 pixels)
- 10 clothing categories
- Pre-split into 60,000 training and 10,000 test samples
- Pixel values normalized from `[0, 255]` → `[0.0, 1.0]`

| Label | Class |
|---|---|
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

---

## 🧠 Models

### Model 1 — Simple (Baseline)

```
Input (28×28) → Flatten → Dense(64, ReLU) → Dense(10, Softmax)
```

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Learning Rate | 0.0005 |
| Batch Size | 32 (default) |
| Epochs | 20 |
| Validation Split | 20% |

---

### Model 2 — Deep (with Regularization)

```
Input (28×28) → Flatten → Dense(256, ReLU) → Dropout(0.3)
             → Dense(128, ReLU) → Dense(64, ReLU) → Dense(10, Softmax)
```

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam |
| Learning Rate | 0.0005 |
| Batch Size | 64 |
| Epochs | 20 |
| Validation Split | 20% |
| Regularization | Dropout (30%) |

---

## 📊 Evaluation & Visualizations

### 1. Training Error Visualization (Loss & Accuracy)

Two side-by-side plots compare the simple and deep models across all epochs:

- **Validation Loss** — how well each model generalizes (lower is better)
- **Validation Accuracy** — classification performance on the held-out set

### 2. Confusion Matrix

A 10×10 heatmap showing predicted vs. true labels for the deep model on the test set. Helps identify which clothing categories are most often confused with each other.

### 3. AUC Score (One-vs-Rest)

The **Area Under the ROC Curve** is computed using a One-vs-Rest strategy for multiclass classification:

```
Deep Model AUC Score: ~0.99xx
```

AUC close to 1.0 indicates excellent class separation ability across all 10 categories.

---

## ⚙️ Hyperparameter Comparison

| Parameter | Simple Model | Deep Model |
|---|---|---|
| Architecture | 1 hidden layer | 3 hidden layers |
| Units | 64 | 256 → 128 → 64 |
| Dropout | None | 0.3 (after first layer) |
| Learning Rate | 0.0005 | 0.0005 |
| Batch Size | 32 | 64 |
| Epochs | 20 | 20 |

A larger batch size (64 vs 32) in the deep model leads to faster training per epoch, while Dropout helps prevent overfitting in the deeper architecture.

---

## 🛠️ Requirements

```bash
pip install tensorflow matplotlib numpy scikit-learn seaborn
```

| Library | Purpose |
|---|---|
| `tensorflow` / `keras` | Model building & training |
| `numpy` | Array operations |
| `matplotlib` | Loss & accuracy plots |
| `seaborn` | Confusion matrix heatmap |
| `scikit-learn` | Confusion matrix, classification report, AUC |

---

## ▶️ How to Run

```bash
python fashion_mnist_classifier.py
```

The script will:
1. Download and normalize the Fashion MNIST dataset
2. Train both models (this may take a few minutes)
3. Display validation loss & accuracy comparison plots
4. Display the confusion matrix heatmap
5. Print the AUC score to the console

---

## 📁 Project Structure

```
├── fashion_mnist_classifier.py   # Main script
└── README.md                     # This file
```

---

## 📈 Expected Results

| Model | Val Accuracy (approx.) | AUC (approx.) |
|---|---|---|
| Simple | ~87% | — |
| Deep | ~89–91% | ~0.99 |

*Results may vary slightly between runs due to random weight initialization.*

---

## 📝 Notes

- The deep model uses **Dropout** as a regularization technique to reduce overfitting
- AUC is computed using **One-vs-Rest (OvR)** strategy, which is standard for multiclass problems
- Both models use the **Adam optimizer** with a tuned learning rate of `0.0005`


