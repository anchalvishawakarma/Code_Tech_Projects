# Decision Tree Classification - Iris Dataset 🌸

## 📌 Objective
This project demonstrates how to build and visualize a **Decision Tree Classification Model** using the **Iris dataset** in Python with **Scikit-learn**.

---

## 🔧 Tools and Libraries Used

- Python
- Pandas, NumPy
- Matplotlib, Seaborn
- Scikit-learn (DecisionTreeClassifier, metrics, train_test_split)

---

## 📊 Dataset Overview

The [Iris dataset](https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html) contains 150 samples of iris flowers with the following features:

- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

Target variable:
- Species (Setosa, Versicolor, Virginica)

---

## 📈 Project Workflow

1. **Data Loading and Exploration**
   - Loaded dataset using `sklearn.datasets.load_iris()`
   - Viewed structure, summary statistics, and sample data

2. **Data Visualization**
   - Distribution plots (Histogram with KDE)
   - Boxen plots to identify outliers

3. **Preprocessing**
   - Feature Scaling using `MinMaxScaler`
   - Train-test split (70:30)

4. **Model Building**
   - Used `DecisionTreeClassifier` from Scikit-learn
   - Trained on scaled data

5. **Model Visualization**
   - Used `plot_tree()` to display splits and leaf nodes

6. **Evaluation**
   - Accuracy score
   - Confusion matrix
   - Classification report (Precision, Recall, F1)

---

## ✅ Results

- Achieved **high accuracy** on the test set
- Model correctly classified most flower types

---

## 📁 Files Included

- `decision_tree_iris.ipynb` – Main code notebook
- `README.md` – Project overview and instructions

---

## 📜 Certificate

Completion certificate will be issued by **CodTech** upon successful submission of this project.

---

## 👩‍💻 Author

**Anchal Vishwakarma**  
*Data Analytics & Machine Learning Enthusiast*

---

## 🔚 End of Project
