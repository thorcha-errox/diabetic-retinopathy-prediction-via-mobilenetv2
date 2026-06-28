# 👁️ RetinAI: Diabetic Retinopathy Intelligence System

[![Live Demo](https://img.shields.io/badge/Demo-Hugging%20Face-blue?style=for-the-badge&logo=huggingface)](https://huggingface.co/spaces/ErrorFree/RetinAI-Diagnostic)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-FF6F00?style=for-the-badge&logo=tensorflow)](https://tensorflow.org)
[![Gradio](https://img.shields.io/badge/UI-Gradio-orange?style=for-the-badge)](https://gradio.app)

An AI-powered diagnostic tool designed to detect and classify stages of **Diabetic Retinopathy (DR)** from retinal fundus images using deep transfer learning.


## 🔗 Live Demo
**Experience the dashboard live here:** [👉 **Click to View RetinAI**](https://huggingface.co/spaces/ErrorFree/RetinAI-Diagnostic)

## 📜 Table of Contents
1. [Project Overview](#1-project-overview)
2. [Key Features](#2-key-features)
3. [Visual Interface](#3-visual-interface)
4. [Tech Stack](#4-tech-stack)
5. [Dataset, Training & Accuracy](#5-dataset-training--accuracy)
6. [Installation Setup](#6-installation-setup)
7. [Medical Disclaimer](#7-medical-disclaimer)

## 📌 Project Overview
Diabetic Retinopathy is a leading cause of blindness globally. Early detection is critical for prevention. This project leverages the **MobileNetV2** architecture to provide a fast, accurate, and accessible classification system for medical professionals and researchers.

### Key Features:
* **Multi-class Classification:** Categorizes scans into 5 stages: No DR, Mild, Moderate, Severe, and Proliferative.
* **Premium Web Interface:** A sleek, user-friendly UI built with Gradio for seamless image uploads.
* **Edge-Ready Architecture:** Uses MobileNetV2 for a lightweight yet powerful inference, making it suitable for low-resource environments.
* **Real-time Analysis:** Get results and confidence scores in seconds.

## 📸 Visual Interface
To showcase the user journey and the system's analytical capabilities, I have included the following interface captures:

### 1. System Landing Page
The landing page is designed with a "Patient-First" approach, focusing on a clean, medical-grade aesthetic that minimizes user friction.
![RetinAI Landing Page](./assets/LandingPage.png)
* This screenshot illustrates the initial state of the application. It highlights the intuitive "Drag & Drop" zone and the clear layout that prepares the user for a seamless diagnostic experience.

### 2. Prediction & Analysis Interface
Once a retinal scan is uploaded, the AI processes the image to provide a multi-class probability breakdown.
![RetinAI Prediction Interface](./assets/PredictionInterface.png)
* This screenshot demonstrates the core intelligence of the system in action. After analyzing the fundus scan, the model classifies the condition (e.g., "Moderate") and provides a detailed confidence bar chart. This transparency allows practitioners to see the model's certainty across all five potential stages.

## 🏗️ Technical Stack
* **Deep Learning Framework:** TensorFlow / Keras
* **Base Model:** MobileNetV2 (Transfer Learning)
* **UI/Deployment:** Gradio & Hugging Face Spaces
* **Language:** Python 3.10
* **Image Processing:** PIL (Pillow), NumPy


## 🔬 Dataset, Training & Accuracy

The model is built using transfer learning with the **MobileNetV2** architecture (pre-trained on ImageNet) and trained on the **APTOS 2019 Blindness Detection** dataset to classify diabetic retinopathy into 5 distinct stages.

### 📊 Dataset Breakdown
* **Source:** APTOS 2019 Blindness Detection dataset (high-resolution retinal fundus photography).
* **Total Samples:** 3,296 images
  * **Training Set:** 2,930 images (89%)
  * **Validation Set:** 366 images (11%)
* **Classes (5 stages of DR):**
  * `0` - No DR
  * `1` - Mild
  * `2` - Moderate
  * `3` - Severe
  * `4` - Proliferative DR

### ⚙️ Training Configuration
* **Data Preprocessing & Augmentation:**
  * Images resized to **224x224** pixels.
  * Rescaled/Normalized pixel values to a **0–1 range**.
  * Augmentation parameters: zoom range of `0.15`, horizontal flips enabled, and `nearest` fill mode.
* **Model Parameters:**
  * **Total Parameters:** 2,422,597 (~9.24 MB)
  * **Trainable Parameters:** 164,613 (custom classifier head)
  * **Non-Trainable Parameters:** 2,257,984 (frozen MobileNetV2 backbone)
* **Optimization:**
  * **Optimizer:** Adam (learning rate = `0.0001`)
  * **Loss Function:** Categorical Cross-entropy
  * **Epochs:** 15 (with `ModelCheckpoint` saving the best validation loss weights)

### 📈 Training Results & Model Accuracy
During training, the model's accuracy and validation performance steadily improved. The final evaluated checkpoint on the validation set yielded:
* **Validation Loss:** `0.6649`
* **Validation Accuracy:** `76.23%` (best overall validation loss epoch)
* **Peak Validation Accuracy:** `76.78%` (achieved at Epoch 12 with a validation loss of `0.6793`)

#### Training Progress Overview:
| Epoch | Training Loss | Training Accuracy | Validation Loss | Validation Accuracy |
| :---: | :-----------: | :---------------: | :-------------: | :-----------------: |
| 1     | 1.7195        | 38.44%            | 0.9018          | 66.39%              |
| 2     | 0.9783        | 66.32%            | 0.8030          | 71.31%              |
| 5     | 0.7899        | 70.81%            | 0.7359          | 72.40%              |
| 10    | 0.7026        | 75.38%            | 0.6920          | 75.41%              |
| 12    | 0.6829        | 76.05%            | 0.6793          | **76.78%**          |
| 15    | 0.6761        | 74.91%            | **0.6649**      | **76.23%** (Saved)  |


## 🛠️ Installation Setup

To run this project locally for development or testing, follow these steps:

### 1. Prerequisites
Ensure you have **Python 3.10** installed on your system.

### 2. Clone the Repository
```bash
git clone https://github.com/Thorcha-Errox/Diabetic-Retinopathy-Project.git
cd Diabetic-Retinopathy-Project
```
### 3. Setup Virtual Environment
```Bash
python -m venv dr_env
# Activate on Windows:
dr_env\Scripts\activate
# Activate on Mac/Linux:
source dr_env/bin/activate
```
### 4. Install Dependencies
```Bash
pip install -r requirements.txt
```
### 5. Launch the Application
```Bash
python app.py
```
Open the provided local URL (usually http://127.0.0.1:7860) in your browser.

## 🏥 Medical Disclaimer
This software is intended for research and educational demonstration purposes only. It does not constitute medical advice or a clinical diagnosis. All AI predictions should be validated by a certified medical professional.

If you find this project helpful, please consider giving it a ⭐ on GitHub!
