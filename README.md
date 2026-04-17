# 🛡️ RNN Email Spam Classifier

A Bidirectional LSTM model built with TensorFlow/Keras for binary email spam classification. Trained on 3,000 emails, the model achieves **98.67% accuracy** and an **AUC-ROC of 0.9998**.

🔗 **[Live Demo](https://noga-66.github.io/RNN-Email-Spam-Classifier/)**

---
<img width="510" height="745" alt="Screenshot 2026-04-17 214238" src="https://github.com/user-attachments/assets/7e4f9533-9f10-4ea6-9216-b522d44871f2" />

## 📊 Results

| Metric | Value |
|---|---|
| Accuracy | **98.67%** |
| AUC-ROC | **0.9998** |
| Precision (Spam) | 0.9600 |
| Recall (Spam) | 0.9600 |
| F1-Score (Spam) | 0.9600 |

---

## 🗂️ Dataset

| Property | Detail |
|---|---|
| File | `spam_or_not_spam.csv` |
| Total emails | 3,000 |
| Ham (not spam) | 2,500 |
| Spam | 500 |
| Class ratio | 5 : 1 (Ham : Spam) |

The class imbalance is handled during training using computed **class weights**, penalising missed spam predictions proportionally.

---

## 🏗️ Model Architecture

```
Input (150 tokens)
    ↓
Embedding (8,000 vocab × 32 dims)
    ↓
SpatialDropout1D (20%)
    ↓
Bidirectional LSTM (32 units × 2 directions = 64 total)
    ↓
GlobalMaxPooling1D
    ↓
Dense (32 units, ReLU)
    ↓
Dropout (30%)
    ↓
Dense (1 unit, Sigmoid) → P(spam)
```

---

## ⚙️ Hyperparameters

| Parameter | Value |
|---|---|
| Vocabulary size | 8,000 |
| Max sequence length | 150 tokens |
| Embedding dimension | 32 |
| LSTM units (per direction) | 32 |
| Batch size | 128 |
| Max epochs | 15 |
| Optimizer | Adam |
| Loss | Binary Crossentropy |

---

## 🚀 Getting Started

### 1. Install dependencies

```bash
pip install tensorflow scikit-learn pandas numpy matplotlib
```

### 2. Prepare the dataset

Place `spam_or_not_spam.csv` in the project root. The file must contain two columns: `email` (text) and `label` (0 = ham, 1 = spam).

### 3. Run the notebook

```bash
jupyter notebook RNN_Spam_Classifier.ipynb
```

Run all cells in order. The notebook will train the model, evaluate it, and save the outputs.

---

## 📁 Output Files

After running the notebook, two files are saved to the project root:

| File | Description |
|---|---|
| `rnn_spam_model.keras` | Trained Keras model |
| `tokenizer_word_index.json` | Tokenizer vocabulary for inference |

---

## 🔍 Live Inference

Use the `predict_email()` function defined in the notebook to classify any email:

```python
predict_email("Congratulations! You've won a FREE iPhone. Click now to claim your prize!")
# 🔴 SPAM  [████████████████████░░░░░░░░░░]  85.3%

predict_email("Hey, are you free for a quick call tomorrow at 2pm?")
# ✅ HAM   [░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░]  3.1%
```

---

## 📈 Training Details

The model uses two callbacks during training:

- **EarlyStopping** — monitors `val_auc`, restores best weights, patience of 4 epochs.
- **ReduceLROnPlateau** — halves the learning rate when `val_loss` plateaus for 2 epochs.

The dataset is split into **80% train / 10% validation / 10% test**, with stratification to preserve the class ratio across all splits.

---

## 📋 Requirements

```
tensorflow >= 2.x
scikit-learn
pandas
numpy
matplotlib
```

---

## 📄 License

This project is for educational and research purposes.
