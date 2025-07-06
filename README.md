# 📦 E-Waste Classification with EfficientNetV2S

This project implements an image classification system to identify different categories of e-waste using transfer learning with EfficientNetV2S in TensorFlow and Google Colab.

---

## 🚀 Project Overview

E-waste management is critical for environmental sustainability. This project classifies electronic waste items (like batteries, keyboards, microwaves, etc.) into 10 categories using deep learning.

We trained a convolutional neural network (CNN) using transfer learning on ~3000 labeled images. The model is also deployed as an interactive web app using Gradio.

---

## 🧪 Model Architecture

✅ **Base Model:** EfficientNetV2S (pretrained on ImageNet)  
✅ **Top Layers:** GlobalAveragePooling2D, Dropout, Dense softmax classifier  
✅ **Data Augmentation:** Random flip, rotation, zoom  
✅ **Optimizer:** Adam  
✅ **Loss:** SparseCategoricalCrossentropy

---

## 🎯 Transfer Learning Strategy

We used a **two-phase training approach**:

### Phase 1: Feature Extraction
- The base EfficientNetV2S model was **frozen**.
- Only the new classification head was trained.
- Used moderate learning rate (1e-3).

### Phase 2: Fine-Tuning
- Unfroze top layers of the base model.
- Trained end-to-end with **lower learning rate (1e-5)**.
- Used EarlyStopping and learning rate scheduling to prevent overfitting.

---

## ✅ Improvements Over Baseline
- Upgraded from **EfficientNetV2B0 → EfficientNetV2S** for better accuracy and efficiency.
- Corrected preprocessing with **`preprocess_input`** for EfficientNetV2.
- Applied **data augmentation** to reduce overfitting.
- Added **Dropout** for regularization.
- Used **EarlyStopping** and **ReduceLROnPlateau** to optimize training.

---

## 📊 Dataset

- ~3000 images
- 10 e-waste categories:
  - Battery
  - Keyboard
  - Microwave
  - Mobile
  - Mouse
  - PCB
  - Player
  - Printer
  - Television
  - Washing Machine

Dataset was split into **train**, **validation**, and **test** folders.

---

## 🖥️ Deployment

The trained model is deployed with a **Gradio** web interface:

✅ Upload an image of an electronic item  
✅ Get predicted class and confidence score

---

## ⚙️ Usage

1. Clone the repository
2. Install requirements
3. Train or load the pretrained model
4. Launch the Gradio app

