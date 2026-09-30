# DS605 Lab 06 — Feature Extraction and Machine Learning

This lab demonstrates feature extraction and traditional machine learning techniques for **image and text data**.

## 📌 Parts Covered

### Part A — Image Classification
- Dataset: Asphalt Crack Dataset (400 images)
- Images resized to 224 × 224 and converted to grayscale.
- Extracted intensity-based features such as:
  - Mean brightness
  - Contrast
  - Dark/bright pixel ratio
  - Intensity range
  - Edge count and edge density
- Canny Edge Detection was used for edge extraction.
- Models:
  - Logistic Regression
  - Decision Tree
  - Random Forest
- Evaluation: Accuracy, Precision, Recall, F1 Score and Confusion Matrix.

**Best observed image result:** Random Forest — **95.83% Accuracy, 95.80% F1 Score**

### Part B — Email Classification
- Dataset: Email Spam Classification Dataset
- 5,172 samples with 3,000 pre-vectorized word-level features.
- Models:
  - Logistic Regression
  - Multinomial Naive Bayes
- Evaluation included classification metrics and training/prediction time.

**Logistic Regression:** 98.00% Accuracy, 96.57% F1 Score

> The provided email dataset was already vectorized, so raw-text processing, CountVectorizer and TF-IDF could not be performed.

### Part C — Representation Improvement
Standardization was applied to the extracted image features before Logistic Regression.

| Representation | Accuracy | F1 Score |
|---|---:|---:|
| Original Features | 78.33% | 79.03% |
| Standardized Features | 94.17% | 94.21% |

Standardization substantially improved Logistic Regression performance by putting features on comparable numerical scales.

## 🛠️ Technologies Used

- Python
- NumPy
- Pandas
- OpenCV
- PIL
- Scikit-learn
- Matplotlib
- Jupyter Notebook

## 📁 Files

- `lab06_image_text.ipynb` — Complete implementation
- `image_features.csv` — Extracted image features
- `README.md` — Project documentation

## 📊 Workflow

**Raw Data → Preprocessing → Feature Extraction/Representation → Train-Test Split → ML Model → Evaluation → Improvement**
