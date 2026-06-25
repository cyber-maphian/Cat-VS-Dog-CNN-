# Cat-VS-Dog-CNN-
# Cat vs Dog Image Classification using Custom CNN (PyTorch)

An end-to-end deep learning pipeline that classifies Cat and Dog images with state-of-the-art accuracy. Built using PyTorch leveraging CNN architecture.

## 🚀 Performance Highlights
* **Validation Accuracy:** `100.00%`
* **Test Accuracy:** `92.00%`

---

## 📌 Project Overview
This repository contains the source code, training pipeline, and evaluation scripts for a Binary-class image classifier trained to recognize:
* Cat
* Dog

---

## 📊 Dataset Structure
The dataset is structured for easy loading via PyTorch's `ImageFolder`:

```text
dataset/
│
├── test
    ├──Cat
    ├──Dog
├── train
    ├──Cat
    ├──Dog
├── val
    ├──Cat
    ├──Dog
