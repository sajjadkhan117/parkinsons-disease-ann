# 🧠 Parkinson's Disease Early Detection Using ANN

![Python](https://img.shields.io/badge/Python-3.10-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13-orange)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red)
![Accuracy](https://img.shields.io/badge/Accuracy-95%25-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

> Early detection of Parkinson's Disease using voice biomarker 
> analysis with a Multi-Layer Artificial Neural Network (ANN).

---

## 🎯 Problem Statement

Parkinson's Disease affects **10+ million people worldwide**.
Current diagnosis happens only after physical symptoms appear
— at an already **ADVANCED stage**.

This project detects Parkinson's **EARLY** from voice recordings
using a trained ANN — before visible symptoms emerge.

---

## 💡 How It Works

```
Voice Features (23)  →  ANN Model  →  Parkinson's / Healthy
```

Parkinson's Disease causes subtle voice abnormalities:
- **High Jitter** — irregular pitch variation
- **High Shimmer** — unstable voice volume  
- **Low HNR** — more noise in voice
- **High PPE** — pitch period irregularity

These changes are **invisible to human ears** but 
**detectable by a trained ANN** with ~95% accuracy!

---

## 🧠 ANN Architecture

```
Input Layer    →  23 voice features
Hidden Layer 1 →  128 neurons | ReLU | Dropout(0.3)
Hidden Layer 2 →  64 neurons  | ReLU | BatchNorm
Hidden Layer 3 →  32 neurons  | ReLU | Dropout(0.2)
Output Layer   →  1 neuron    | Sigmoid
```

---

## 📊 Dataset

| Property | Details |
|---|---|
| Source | UCI ML Repository |
| Researcher | Max Little, Oxford University |
| Total Samples | 195 voice recordings |
| Parkinson's | 147 patients (75.4%) |
| Healthy | 48 subjects (24.6%) |
| Features | 23 biomedical voice measurements |

---

## 📈 Results

| Metric | Score |
|---|---|
| Test Accuracy | ~95% |
| ROC-AUC Score | >0.95 |
| Precision | ~96% |
| F1 Score | ~97% |

---

## 🖥️ Screenshots

### Parkinson's Detected 🔴
> (Add your app screenshot here)

### Healthy Detected 🟢  
> (Add your app screenshot here)

---

## 🚀 Installation & Run

### 1. Clone Repository
```bash
git clone https://github.com/YourUsername/parkinsons-ann.git
cd parkinsons-ann
```

### 2. Create Environment
```bash
conda create -n parkinsons_ann python=3.10
conda activate parkinsons_ann
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Train Model
```bash
python model.py
```

### 5. Run Web App
```bash
streamlit run app.py
```

### 6. Open Browser
```
http://localhost:8501
```

---

## 📁 Project Structure

```
parkinsons-ann/
│
├── model.py              ← ANN training script
├── app.py                ← Streamlit web app
├── parkinsons.csv        ← Dataset
├── requirements.txt      ← Dependencies
├── README.md             ← Documentation
│
├── plots/
│   ├── confusion_matrix.png
│   ├── training_history.png
│   ├── roc_curve.png
│   ├── feature_importance.png
│   ├── correlation_heatmap.png
│   └── class_distribution.png
```

---

## 🛠️ Tech Stack

- **Python 3.10** — Core language
- **TensorFlow/Keras** — ANN model
- **Pandas & NumPy** — Data processing
- **Scikit-learn** — Preprocessing & metrics
- **Matplotlib & Seaborn** — Visualizations
- **Streamlit** — Web application

---

## 🌍 Real World Impact

```
😔 Before: Late-stage diagnosis, limited treatment options
😊 After:  Early voice-based detection, better outcomes
```

This tool can assist neurologists with **non-invasive,
low-cost, early-stage Parkinson's screening**.

---

## 👥 Team

| Name | Role |
|---|---|
| Member 1 | Model Development |
| Member 2 | Web App Development |
| Member 3 | Data Analysis & Visualization |

---

## 📚 References

- [UCI Parkinson's Dataset](https://archive.ics.uci.edu/dataset/174/parkinsons)
- Max A. Little et al. — *Suitability of Dysphonia Measurements 
  for Telemonitoring of Parkinson's Disease*
- TensorFlow Documentation
- Streamlit Documentation

---

## 📄 License

This project is licensed under the **MIT License**.

---

⭐ **If you found this useful, please give it a star!** ⭐
