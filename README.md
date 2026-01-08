# 📊 Optimizer & Activation Function Benchmarking for Regression (PyTorch)

## Overview

This repository contains a **Google Colab–ready PyTorch notebook** that benchmarks the impact of **different optimizers and activation functions** on a **regression task**. The notebook is designed as a reproducible experiment to analyze how training dynamics and generalization performance vary with architectural and optimization choices.

The project demonstrates strong experimentation practices, including clean data splits, early stopping, consistent training settings, and clear visualization of results. It is well-suited for **GitHub showcasing, interviews, and educational purposes**.

---

## 🎯 Purpose of the Notebook

When training neural networks, choices such as **optimizer** and **activation function** can significantly influence:

- Convergence speed
- Training stability
- Final generalization performance

Instead of relying on default configurations, this notebook systematically evaluates multiple combinations using:
- A fixed model architecture
- Identical train/validation/test splits
- Identical training budgets and stopping criteria

The notebook helps answer questions such as:
- Which activation functions converge faster?
- Which optimizers generalize better on unseen data?
- How do learning curves differ across configurations?

---

## 📂 Dataset

- **California Housing Dataset** (public, via `scikit-learn`)
- Task type: Regression
- Target: Median house value
- Features: Demographic and geographic attributes

### Data Split Strategy
- **70% Training**
- **15% Validation**
- **15% Test**

Feature scaling is applied using `StandardScaler`, fitted only on the training set to avoid data leakage.

---

## 🧠 Model Architecture

A lightweight **Multi-Layer Perceptron (MLP)** is used to keep the focus on optimizer and activation behavior rather than architectural complexity.

**Architecture:**
- Input layer (feature dimension)
- Hidden Layer 1: 128 units
- Hidden Layer 2: 64 units
- Output Layer: 1 unit (regression)

The activation function is configurable and injected dynamically during model creation.

---

## ⚙️ Experiment Design

### Activation Functions Evaluated
- ReLU
- Tanh
- GELU
- LeakyReLU

### Optimizers Evaluated
- SGD (with momentum)
- Adam
- RMSprop
- AdamW

### Training Details
- Loss function: Mean Squared Error (MSE)
- Early stopping based on validation loss
- Best model checkpoint restored before evaluation
- GPU acceleration supported (Google Colab)

---

## 📈 Evaluation Metrics

Since this is a regression task, **classification accuracy is not applicable**.

The following metrics are used:

- **RMSE (Root Mean Squared Error)**  
  Lower values indicate better performance.

- **R² Score (Coefficient of Determination)**  
  Used as a proxy for “accuracy” in regression.  
  Higher values indicate better generalization.

---

## 📊 Visualizations Included

The notebook generates the following plots:

1. **Learning Curves by Activation Function**  
   - Validation loss averaged across optimizers  
   - Highlights convergence behavior and stability

2. **Learning Curves by Optimizer**  
   - Validation loss averaged across activation functions  
   - Shows optimizer effectiveness and robustness

3. **Test Performance Comparison**
   - Bar charts of average R² by activation
   - Bar charts of average R² by optimizer

4. **Best Model Learning Curve**
   - Training vs validation loss
   - Useful for diagnosing overfitting or underfitting

---

## ▶️ Usage Example

### Run in Google Colab

1. Open Google Colab
2. Upload the notebook or clone the repository:
3. Open the notebook and select Runtime → Run all
4. Results and plots will be generated automatically

**Local Execution (Optional)**

 ```bash
pip install torch scikit-learn numpy pandas matplotlib
jupyter notebook
  
