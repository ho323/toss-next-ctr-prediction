# TOSS NEXT CTR Prediction
<img width="1877" height="382" alt="image" src="https://github.com/user-attachments/assets/18644147-6326-4a2e-b444-44efe4b270cb" />  

A click-through rate prediction project for the **TOSS NEXT ML Challenge**, using deep learning and tree-based models for tabular recommendation data.

This repository implements and compares multiple CTR prediction models, including **HDCN**, **Hybrid GDCN**, and **XGBoost**, and provides an ensemble-oriented inference workflow for competition submission.

---

## Overview

Click-through rate prediction is a core problem in recommendation systems and online advertising.
The goal is to estimate the probability that a user will click on a given item or advertisement based on structured user, item, and context features.

This project was developed for the **TOSS NEXT ML Challenge** hosted on Dacon.

Competition link:
https://dacon.io/competitions/official/236575/overview/description

The project focuses on:

* CTR prediction with structured tabular data
* Deep learning-based feature interaction modeling
* Graph-inspired cross-feature modeling
* Tree-based boosting baseline
* Model comparison and ensemble inference

---

## Key Features

* HDCN-based CTR prediction model
* Hybrid GDCN model for feature interaction learning
* XGBoost baseline and ensemble component
* Model weight management for inference
* Submission-ready prediction output generation
* Notebook-based experimentation workflow
* PyTorch and XGBoost-based model training

---

## Models

### 1. HDCN: Hybrid Deep Cross Network

HDCN is a deep learning-based cross network model designed to capture interactions between categorical and numerical features.

Main characteristics:

* Tabular CTR modeling
* Feature interaction learning
* Deep neural network-based representation learning
* PyTorch implementation

---

### 2. Hybrid GDCN: Hybrid Graph Deep Cross Network

Hybrid GDCN combines cross-network style feature interaction modeling with graph-inspired representation learning.

Main characteristics:

* Hybrid deep learning architecture
* Graph-based feature interaction modeling
* CTR-oriented prediction structure
* Notebook-based training and inference workflow

---

### 3. XGBoost

XGBoost is used as a strong tree-based baseline and ensemble component.

Main characteristics:

* Gradient boosting-based tabular modeling
* Strong performance on structured data
* Fast training and inference
* Useful baseline for deep CTR models

---

## Repository Structure

```
.
├── HDCN/
│   ├── hdcn_train.py              # HDCN training script
│   └── hdcn_inference.py          # HDCN inference script
│
├── Hybrid_GDCN/
│   ├── Hybrid_GDCN_train.ipynb    # Hybrid GDCN training notebook
│   ├── Hybrid_GDCN_inference.ipynb
│   ├── basic_layers.py
│   └── model_hybrid_gdcn_5epch.pt
│
├── XGB/
│   ├── train.ipynb                # XGBoost training notebook
│   ├── inference.ipynb            # XGBoost inference notebook
│   └── xgb_model.json
│
├── models_weight/
│   ├── hdcn_noseq.pth
│   ├── model_hybrid_gdcn_5epch.pt
│   └── xgb_model.json
│
├── output/
│   ├── hdcn.csv
│   ├── hybrid_gdcn.csv
│   └── xgb_infer.csv
│
├── base.py                        # Base abstraction module
├── main.ipynb                     # Main experiment notebook
├── inference.ipynb                # Integrated inference notebook
├── environment.ipynb              # Environment setup notebook
├── requirements.txt
├── sample_submission.csv
└── README.md
```

---

## Installation

Clone the repository:

```
git clone https://github.com/ho323/toss-next-ctr-prediction.git
cd toss-next-ctr-prediction
```

Create and activate a virtual environment:

```
python -m venv venv
```

For Windows:

```
venv\Scripts\activate
```

For Linux or macOS:

```
source venv/bin/activate
```

Install dependencies:

```
pip install -r requirements.txt
```

PyTorch may need to be installed separately depending on your CUDA version.

Example for CUDA 12.1:

```
pip install torch==2.3.1+cu121 --index-url https://download.pytorch.org/whl/cu121
```

---

## Main Dependencies

| Package      | Version     |
| ------------ | ----------- |
| numpy        | 1.26.4      |
| pandas       | 2.3.2       |
| scikit-learn | 1.6.1       |
| xgboost      | 3.0.5       |
| lightgbm     | 4.6.0       |
| catboost     | 1.2.8       |
| optuna       | 4.5.0       |
| torch        | 2.3.1+cu121 |

---

## Data Preparation

Place the competition dataset under the `data/` directory.

Expected files:

```
data/
├── train.parquet
└── test.parquet
```

The dataset is not included in this repository due to competition and data-sharing restrictions.

---

## Training

### 1. Train HDCN

```
cd HDCN
python hdcn_train.py
```

### 2. Train Hybrid GDCN

Run the notebook:

```
Hybrid_GDCN/Hybrid_GDCN_train.ipynb
```

### 3. Train XGBoost

Run the notebook:

```
XGB/train.ipynb
```

---

## Inference

### 1. HDCN Inference

```
cd HDCN
python hdcn_inference.py
```

### 2. Hybrid GDCN Inference

Run the notebook:

```
Hybrid_GDCN/Hybrid_GDCN_inference.ipynb
```

### 3. XGBoost Inference

Run the notebook:

```
XGB/inference.ipynb
```

### 4. Integrated Inference

Run one of the integrated notebooks:

```
inference.ipynb
main.ipynb
```

---

## Output Files

Prediction results are saved under the `output/` directory.

```
output/
├── hdcn.csv
├── hybrid_gdcn.csv
└── xgb_infer.csv
```

These outputs can be combined or used for final submission depending on the experiment setup.

---

## Experiment Configuration

Example HDCN configuration:

```
CFG = {
    "BATCH_SIZE": 256,
    "EPOCHS": 5,
    "LEARNING_RATE": 1e-3,
    "SEED": 42
}
```

Hyperparameters can be modified in each model script or notebook.

---

## Project Relevance

This project is related to:

* Click-through Rate Prediction
* Recommendation Systems
* Tabular Deep Learning
* Feature Interaction Modeling
* Ensemble Learning
* XGBoost
* PyTorch
* Advertising ML
* Dacon AI Competition

---

## Limitations

* This repository is intended for competition, research, and portfolio purposes.
* The original competition dataset is not included due to data-sharing restrictions.
* Some workflows are notebook-based and may require manual execution.
* Model performance may vary depending on preprocessing, feature engineering, random seed, and validation strategy.
* Additional refactoring would be required for production-level CTR prediction systems.

---

## Future Work

* Refactor notebook-based workflows into reusable Python scripts
* Add unified training and inference CLI
* Add experiment tracking with MLflow or Weights & Biases
* Add validation metrics and leaderboard score summary
* Add ensemble blending script
* Add feature importance and error analysis
* Add reproducible configuration files
* Add Docker-based environment setup

