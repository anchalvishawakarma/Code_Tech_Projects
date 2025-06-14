# 🧠 Customer Review Sentiment Analysis

This project is a **text classification model** built using **TF-IDF** and **Logistic Regression** to analyze customer reviews and predict whether they are **positive** or **negative**.

---

## 📁 Dataset

The dataset used is `customer_reviews.csv`, which includes:
- `Review`: The raw customer review text.
- `Sentiment`: The label (`Positive` or `Negative`).

---

## 🚀 Features

- Text cleaning (lowercasing, punctuation and number removal).
- TF-IDF vectorization.
- Logistic Regression classification.
- Evaluation via accuracy, confusion matrix, and classification report.
- Custom prediction function to test new reviews.

---

## 🛠️ Installation

```bash
pip install pandas numpy scikit-learn matplotlib seaborn
