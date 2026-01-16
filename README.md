# iot-attack-detection-dashboard
"IoT Attack Detection System using ML models (Random Forest, CNN, LSTM) on CICIDS2017 dataset - 99.90% accuracy"

# 🛡️ IoT Attack Detection System

Real-time IoT network attack detection and classification using machine learning models on the CICIDS2017 dataset.

[![Live Demo](https://img.shields.io/badge/demo-live-success)](file:///C:/Users/anany/Downloads/iot_attack_detection_dashboard_UPDATED.html)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## 🎯 Project Overview

This project implements and compares 6 different machine learning models for detecting IoT network attacks with **99.90% accuracy**. The system processes 2M+ network traffic records and provides real-time attack classification with SHAP explainability.

### Key Results
- **Best Model:** Random Forest
- **Accuracy:** 99.90%
- **Inference Time:** 0.726 seconds (34,700 packets/sec)
- **False Positives:** Very Low (5)
- **Dataset:** CICIDS2017 (125,973 samples)

## 🚀 Live Demo

**[View Interactive Dashboard →](C:/Users/anany/Downloads/iot_attack_detection_dashboard_UPDATED.html)**

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
- Random Forest (Winner - 99.90%)
- SVM (98.82%)
- Neural Network (97.52%)
- CNN (95.84%)
- LSTM (86.82%)
- Autoencoder (65.98%)

**Visualization:**
- React (Interactive Dashboard)
- Custom CSS3 Animations
- Responsive Design

## 📊 Features

### Attack Detection
- ✅ 12 attack type classification (DoS, DDoS, Probe, U2R, R2L)
- ✅ Real-time inference (34,700 packets/second)
- ✅ 99.90% accuracy on test set
- ✅ Minimal false positives

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

## 💾 Dataset

**CICIDS2017** - Canadian Institute for Cybersecurity IDS Dataset
- **Total Records:** 125,973
- **Training Set:** 100,778 samples
- **Test Set:** 25,195 samples
- **Features:** 38 network traffic features
- **Classes:** Normal, DoS, DDoS, Probe, U2R, R2L

### Class Distribution
- Normal: 13,469 (53.5%)
- Other (DoS, Probe): 11,706 (46.5%)
- R2L: 12 (0.05%)
- U2R: 8 (0.03%)

## 🎓 Research Impact

This project was conducted at **Syracuse University** as part of graduate research in IoT security and demonstrates:
- Traditional ML superiority over Deep Learning for tabular data
- Importance of model explainability in security applications
- Production-ready ML deployment considerations
- Handling extreme class imbalance in cybersecurity

## 📝 Citation

If you use this work, please cite:
```
@misc{nagaraj2025iot,
  author = {Ananya Naga Raj},
  title = {IoT Attack Detection System using Machine Learning},
  year = {2025},
  publisher = {Syracuse University},
  url = {https://github.com/Ananyanagaraj11/iot-attack-detection}
}
```

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

## 📧 Contact

**Ananya Naga Raj**
- GitHub: [@Ananyanagaraj11](https://github.com/Ananyanagaraj11)
- LinkedIn: [linkedin.com/in/AnanyaNagaRaj](https://linkedin.com/in/AnanyaNagaRaj)
- Email: annagara@syr.edu

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

⭐ Star this repo if you find it helpful!
