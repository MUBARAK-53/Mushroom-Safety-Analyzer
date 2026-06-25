
# 🍄 Mushroom Safety Classification System

A machine learning-based web application that predicts whether a mushroom is **edible or poisonous** using an **XGBoost classifier**. The project is deployed as an interactive **Streamlit dashboard** with a clinical-style user interface.

---

## 📌 Project Overview

This application demonstrates an end-to-end machine learning workflow, from data preprocessing and feature encoding to model inference and interactive visualization.

The system classifies mushrooms based on selected physical characteristics and provides real-time predictions along with confidence scores.

---

## 🎯 Key Features

* Real-time mushroom toxicity prediction
* Binary classification (Edible / Poisonous)
* Confidence score estimation
* Interactive Streamlit dashboard
* Encoded feature inspection
* Tab-based navigation for usability
* Structured ML inference pipeline

---

## 🧪 Input Features

The model uses the following categorical features:

* Odor
* Cap color
* Spore print color
* Bruises
* Stalk root type

These features are encoded into a machine-readable format before being passed to the model.

---

## ⚙️ Tech Stack

* Python
* Streamlit
* XGBoost
* Pandas
* Scikit-learn
* Pickle (for feature encoding)

---

## 🧠 Model Pipeline

1. User inputs mushroom characteristics via UI
2. Input is encoded using saved feature mappings
3. Data is transformed into model-compatible format
4. XGBoost model performs inference
5. Output includes:

   * Predicted class (Edible / Poisonous)
   * Confidence score

---

## 🖥️ Application Interface

The application is structured into three main sections:

* **Analysis Tab** – Input features and predictions
* **Feature Guide** – Description of input variables
* **Model Information** – Details about model and methodology

The interface is designed for clarity, interpretability, and ease of use.

---

## 📁 Project Structure

```
mushroom-ai-analyzer/
│
├── app.py                    # Streamlit application
├── mushroom_model.json      # Trained XGBoost model
├── feature_columns.pkl      # Encoded feature columns
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation
```

## 📦 Requirements

```
streamlit
pandas
xgboost
scikit-learn
pickle-mixin
```

---

## 📊 Model Information

* **Algorithm:** XGBoost Classifier
* **Problem Type:** Binary Classification
* **Output Classes:** Edible, Poisonous
* **Input Type:** Encoded categorical features
* **Objective:** Toxicity prediction based on morphological attributes

---

## ⚠️ Disclaimer

This project is intended for **educational and research purposes only**.

It should not be used as a substitute for professional mycological or medical advice.

---

## 👨‍💻 Author

**Mubarak Naikwade**
GitHub: [MUBARAK-53](https://github.com/MUBARAK-53)

---
