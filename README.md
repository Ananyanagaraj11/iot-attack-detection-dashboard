# 🛡️ IoT Attack Detection System

Real-time IoT network attack detection and classification using machine learning models on the CICIDS2017 dataset.

[![Live Demo](https://img.shields.io/badge/demo-live-success)](https://ananyanagaraj11.github.io/iot-attack-detection-dashboard/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## 🎯 Project Overview

Comprehensive comparison of 6 machine learning models for IoT attack detection, achieving **99.90% accuracy** with Random Forest. Processes 2M+ network traffic records with real-time classification and SHAP explainability.

### Key Results
- **Best Model:** Random Forest
- **Accuracy:** 99.90%
- **Inference Time:** 0.726 seconds (34,700 packets/sec)
- **False Positives:** Very Low (5)
- **Dataset:** CICIDS2017 (125,973 samples)

## 🚀 Live Demo

**[View Interactive Dashboard →](https://ananyanagaraj11.github.io/iot-attack-detection-dashboard/)**

The dashboard includes:
- Comprehensive model comparison (6 ML models)
- SHAP explainability analysis
- Confusion matrices and performance metrics
- Attack type classification (12 categories)
- Real-time inference speed comparison

## 🛠️ Tech Stack

**Machine Learning:**
- Python, PyTorch, TensorFlow
- Scikit-learn, Pandas, NumPy
- SHAP (Model Explainability)

**Models Tested:**
1. **Random Forest** (99.90% - Winner)
2. SVM (98.82%)
3. Neural Network (97.52%)
4. CNN (95.84%)
5. LSTM (86.82%)
6. Autoencoder (65.98%)

**Visualization:**
- React (Interactive Dashboard)
- HTML5, CSS3
- Responsive Design

## 📊 Features

### Attack Detection
- ✅ 12 attack type classification (DoS, DDoS, Probe, U2R, R2L)
- ✅ Real-time inference (34,700 packets/second)
- ✅ 99.90% accuracy on test set
- ✅ Minimal false positives (5 total)

### Explainability
- ✅ SHAP feature importance analysis
- ✅ Top 8 security-relevant features identified
- ✅ Attack pattern visualization
- ✅ Transparent decision-making

### Performance
- ✅ 31x faster than SVM
- ✅ 27x faster than LSTM
- ✅ Production-ready inference speed
- ✅ Lightweight model (100 trees)

## 📈 Model Comparison

| Model | Accuracy | Inference Time | False Positives |
|-------|----------|----------------|-----------------|
| **Random Forest** | **99.90%** | **0.726s** | **5** |
| SVM | 98.82% | 22.894s | 81 |
| Neural Network | 97.52% | 1.491s | 343 |
| CNN | 95.84% | 2.559s | 603 |
| LSTM | 86.82% | 19.878s | 1,437 |
| Autoencoder | 65.98% | 2.642s | Moderate |

## 🔍 SHAP Analysis - Top Features

1. **serror_rate** (8.67%) - SYN error rate, DoS/DDoS indicator
2. **num_file_creations** (8.19%) - File system attack indicator
3. **num_shells** (7.73%) - Shell access attempts
4. **srv_count** (6.71%) - Service connection patterns
5. **num_compromised** (2.85%) - Direct attack success indicator
6. **dst_host_rerror_rate** (2.80%) - Rejection errors
7. **urgent** (2.76%) - Urgent packets indicator
8. **root_shell** (2.74%) - Root access detection

## 💾 Dataset

**CICIDS2017** - Canadian Institute for Cybersecurity IDS Dataset
- **Total Records:** 125,973
- **Training Set:** 100,778 samples
- **Test Set:** 25,195 samples
- **Features:** 38 network traffic features
- **Classes:** Normal, DoS, DDoS, Probe, U2R, R2L

### Class Distribution
- Normal: 13,469 (53.5%)
- Other Attacks: 11,706 (46.5%)
- R2L: 12 (0.05%)
- U2R: 8 (0.03%)

## 🎓 Research Impact

This project demonstrates:
- Traditional ML superiority over Deep Learning for tabular data
- Importance of model explainability in security applications
- Production-ready ML deployment considerations
- Handling extreme class imbalance in cybersecurity

**Developed at Syracuse University** - Graduate Research Project, Fall 2025

## 💻 Usage

Simply open the dashboard:
