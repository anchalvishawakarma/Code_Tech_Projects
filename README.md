# 🚀 CODTECH Internship Projects

This repository contains a collection of **Machine Learning**, **Deep Learning**, and **Recommendation System** tasks completed during the **CODTECH Internship**. Each task is implemented in **Python**, with clear, well-commented code and detailed methodology.

---

## 📌 Table of Contents

- [📂 Task 1: Decision Tree Classification](#-task-1-decision-tree-classification)
- [📂 Task 2: Sentiment Analysis](#-task-2-sentiment-analysis)
- [📂 Task 3: CNN for Image Classification](#-task-3-cnn-for-image-classification)
- [📂 Task 4: Recommendation System](#-task-4-recommendation-system)
- [▶️ How to Run](#️-how-to-run)
- [📦 Dependencies](#-dependencies)
- [🧑‍💻 Author](#-author)
- [📄 License](#-license)

---

## 📂 Task 1: Decision Tree Classification

**🎯 Objective:**  
Build and visualize a **Decision Tree Classifier** using Scikit-learn.

**🗃️ Dataset:**  
[Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality) (~4,000+ records)

**🔍 Key Steps:**
- Data loading and preprocessing
- Model training with `DecisionTreeClassifier`
- Tree visualization using `matplotlib`
- Evaluation (accuracy score)

**📈 Output:**
- Graphical decision tree plot
- Accuracy and prediction metrics

---

## 📂 Task 2: Sentiment Analysis

**🎯 Objective:**  
Classify customer reviews into positive or negative sentiments using **TF-IDF** and **Logistic Regression**.

**🗃️ Dataset:**  
Preprocessed subset of the [Sentiment140 dataset](http://help.sentiment140.com/home)

**🔍 Key Steps:**
- Text cleaning using regular expressions
- Feature extraction via `TfidfVectorizer`
- Model training with `LogisticRegression`
- Evaluation using accuracy, precision, recall, F1-score

**🔮 Example:**

```python
predict_sentiment("The product is amazing!")  # Output: Positive 😊

---

## 📂 Task 3: CNN for Image Classification 

**🎯 Objective:**  
Build a **Convolutional Neural Network (CNN)** using TensorFlow to classify **cat vs dog images**.

**🗃️ Dataset:**  
Custom dataset of cat and dog images (e.g., Kaggle’s ["Dogs vs. Cats"](https://www.kaggle.com/competitions/dogs-vs-cats)).

**🔍 Key Steps:**
- Load and preprocess image data (resize, normalize).
- Create training and validation image generators.
- Build CNN model using `Conv2D`, `MaxPooling2D`, `Flatten`, and `Dense` layers.
- Train the model and validate performance.

**📈 Output:**
- Training/validation accuracy and loss plots.
- Final model accuracy on validation/test data.
- Predictions on new images.

---

## 📂 Task 4: Recommendation System

**🎯 Objective:**  
Build a **movie recommendation system** using **Collaborative Filtering (cosine similarity)**.

**🗃️ Dataset:**  
[MovieLens 100k](https://grouplens.org/datasets/movielens/) dataset (user ratings for movies).

**🔍 Key Steps:**
- Create a user-item interaction matrix.
- Calculate cosine similarity between items.
- Generate top-N movie recommendations for a user.
- Evaluate using **Mean Squared Error (MSE)** or verify relevance manually.

**📈 Output:**
- Personalized movie recommendations.
- Evaluation of similarity metrics.

---

## ▶️ How to Run

**Clone the repository:**
```bash
git clone https://github.com/your-username/codtech-internship-projects.git
cd codtech-internship-projects


Install required libraries:

```bash

pip install -r requirements.txt
Run individual project files:

```bash

python task1_decision_tree.py
python task2_sentiment_analysis.py
python task3_cnn_image_classification.py
python task4_recommendation_system.py

📦 Dependencies
            pandas

            numpy

            matplotlib

            seaborn

            scikit-learn

            tensorflow

Python built-in libraries: re, os

You can install all dependencies at once using:

```bash

pip install -r requirements.txt

🧑‍💻 Author
Anchal Vishwakarma
🎓 MCA | 💼 Data Analyst & ML Enthusiast


