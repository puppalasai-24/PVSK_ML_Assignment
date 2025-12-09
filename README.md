# Understanding the Impact of Activation Functions on Deep Learning  
### A Practical Comparison of ReLU, LeakyReLU, ELU, and GELU Using the Dry Bean Dataset

This repository accompanies the tutorial and report that compare four major activation functions—**ReLU**, **LeakyReLU**, **ELU**, and **GELU**—using a controlled experimental setup on a real-world tabular dataset.  
The goal is to help readers understand *how activation functions influence optimisation dynamics, gradient flow, and final performance* in deep neural networks.

This README corresponds to the report:


---

## 📌 Project Overview

This project demonstrates:

- Why activation functions are essential in neural networks  
- Mathematical intuition behind ReLU and its variants  
- A controlled experiment using **identical deep MLP architectures**  
- Comparison of:
  - Training accuracy  
  - Validation accuracy  
  - Training loss  
  - Validation loss  
  - Final test accuracy  
- A clear, educational explanation for why performance differences were minimal  
- When activation choice *does* become important (deep/complex models)

This tutorial supports key module learning outcomes:  
✔ critical comparison of ML methods  
✔ programming non-trivial ML models  
✔ understanding optimisation and feature interactions  
✔ ability to communicate ML concepts effectively  

---

## 📊 Dataset

**Dataset:** *Dry Bean Dataset*  
Source: Kaggle  
Link: https://www.kaggle.com/datasets/jasonchan2208/dry-bean-dataset  

The dataset contains:

- 16 numeric morphological features  
- 7 bean classes  
- ~13,000 samples  

### Preprocessing Steps

As described in the report (Section 3):  
- Label-encode target classes  
- Standardise numeric features using `StandardScaler`  
- Train/validation/test split: **70% / 15% / 15%**

### ⚠️ Dataset Not Included in Repository  
Kaggle datasets cannot be redistributed via GitHub due to size and licence constraints.

Please download the dataset manually from Kaggle and place it in:


---

## 🧠 Activation Functions Compared

We compare four widely used linear-unit activations:

### **ReLU**
- Fast, simple, sparse
- Can suffer from dying neurons

### **LeakyReLU**
- Prevents dead neurons by allowing a small negative slope

### **ELU**
- Smooth negative region  
- Helps shift activations closer to zero mean

### **GELU**
- Smooth, probabilistic gating  
- Used in Transformers (BERT, GPT)

The report explains the mathematics and intuition behind each function (Section 2) :contentReference[oaicite:1]{index=1}.

---

## 🧪 Experimental Setup

As detailed in Section 4 of the report :contentReference[oaicite:2]{index=2}:

### **Model Architecture (Identical Across All Experiments)**

- Input: 16 standardised features  
- Hidden layers: **3 × Dense(256 units)**  
- Activation: ReLU / LeakyReLU / ELU / GELU  
- Dropout: 0.2  
- Output: 7-class softmax  
- Optimiser: Adam (learning rate = 1e-3)  
- Epochs: 30  
- Batch size: 128  

### **Metrics collected**
- Training accuracy  
- Validation accuracy  
- Training loss  
- Validation loss  
- Final test accuracy  

Figures include:  
- Training accuracy curves  
- Validation accuracy curves  

---

## 🛠️ How to Run the Notebook

### 1. Install dependencies

```bash
pip install numpy pandas scikit-learn tensorflow matplotlib seaborn
```

### 2. Add the dataset manually

Place the downloaded dataset into:

```
data/DryBeanDataset.csv
```
--- 

### 3. Run the notebook
```
PVSK_ML_Assignment.ipynb
```

This will automatically:

- Load and preprocess the dataset

- Train four models

- Generate accuracy and loss plots

- Print final test metrics
---

## 📁 Folder Structure
```
project/
│
├── data/
│   └── DryBeanDataset.csv     # Not included (download from Kaggle)
│
├── figures/
│   ├── training_accuracy_relu.png
│   ├── training_accuracy_leakyrelu.png
│   ├── training_accuracy_elu.png
│   ├── training_accuracy_gelu.png
│   ├── validation_accuracy_relu.png
│   └── ...
│
├── activation_comparison.ipynb
├── PVSK_ML_Report.pdf
├── README.md
└── LICENSE
```
---

## 📈 Results Summary

From the report (Section 6)
```
| Activation    | Test Accuracy |
| ------------- | ------------- |
| **ReLU**      | 0.9236        |
| **LeakyReLU** | 0.9226        |
| **ELU**       | 0.9216        |
| **GELU**      | **0.9241**    |
```
---

## Key Observations

All activations performed extremely similarly (>92%).

- Differences were less than 0.3% — statistically negligible.

- GELU performed marginally best.

- ReLU remained highly competitive despite being simplest.

- Theoretical benefits of ELU and GELU did not translate into major gains on this dataset.
---

## 🧠 Discussion Highlights

From Section 7 of the report 

PVSK_ML_Report:

- Why metrics were similar:

- Well-scaled tabular data reduces activation sensitivity

- Moderate model depth (3 layers)

- Adam optimiser stabilises gradient behaviour

- No long-range dependencies (unlike NLP or vision tasks)

- Positive-region behaviour is similar across activations

Subtle differences (visible in curves):

- ReLU → fastest early convergence

- LeakyReLU → avoids small stalls

- ELU → smoother gradient flow

- GELU → most consistent, stable curves

When activation choice does matter:

- Deep networks

- Transformer models

- GANs

- Unnormalised or skewed data

- Tasks requiring long-range interactions
---

## 🧩 Conclusions

Summarised from Section 8 of the report:

- All four activation functions achieve comparable accuracy.

- GELU performs marginally best, but ReLU is nearly identical.

- Activation effects depend heavily on context and architecture depth.

- Understanding activation functions is essential even when differences are subtle.
---

## 📚 References

As listed in the report (Section 9):

- He et al. (2015) — Delving Deep into Rectifiers

- Clevert et al. (2015) — Exponential Linear Units (ELUs)

- Hendrycks & Gimpel (2016) — Gaussian Error Linear Units (GELUs)

- Goodfellow et al. (2016) — Deep Learning
---

## 📄 License

This project is licensed under the MIT License.
See LICENSE for details.
