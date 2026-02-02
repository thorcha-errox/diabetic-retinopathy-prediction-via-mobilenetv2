# 👁️ RetinAI: Diabetic Retinopathy Intelligence System

[![Live Demo](https://img.shields.io/badge/Demo-Hugging%20Face-blue?style=for-the-badge&logo=huggingface)]([https://huggingface.co/spaces/ErrorFree/RetinAI-Diagnostic])
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
5. [Installation Setup](#5-installation-setup)
6. [Medical Disclaimer](#6-medical-disclaimer)

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


## 🔬 Dataset & Training
The model was trained on high-resolution retinal fundus photography. Data preprocessing included:
* Resizing to **224x224** pixels.
* Normalization (scaling pixel values to a **0-1 range**).
* Data augmentation to improve model robustness against variations in lighting and camera quality.


## 🛠️ Installation Setup

To run this project locally for development or testing, follow these steps:

### 1. Prerequisites
Ensure you have **Python 3.10** installed on your system.

### 2. Clone the Repository
```bash
git clone [https://github.com/Thorcha-Errox/Diabetic-Retinopathy-Project.git](https://github.com/Thorcha-Errox/Diabetic-Retinopathy-Project.git)
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

## 👤 Author
Your Name AI Enthusiast & Developer Your GitHub Profile | Your LinkedIn

If you find this project helpful, please consider giving it a ⭐ on GitHub!
