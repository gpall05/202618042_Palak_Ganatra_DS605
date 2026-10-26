# Lab Assignment 8: Decision Trees and Support Vector Machines

## Overview

This assignment explores Decision Trees and Support Vector Machines (SVMs) through implementation, visualization, and comparison. The main aim was to understand how these algorithms make predictions and how their performance changes with different parameters and kernel functions.

The assignment is divided into four parts, covering manual implementations, Scikit-learn models, nonlinear classification, and a final performance comparison.

## Dataset

### 1. Banknote Authentication Dataset

The Banknote Authentication dataset from the UCI Machine Learning Repository is used for Decision Tree and SVM experiments.

It contains 1,372 samples with four input features:

- Variance
- Skewness
- Curtosis
- Entropy

The target variable represents the class of the banknote.

Two features, variance and skewness, are used for visualizing decision boundaries. All four features are used for the final model comparison.

### 2. Two-Moons Dataset

The two-moons dataset is generated using Scikit-learn's `make_moons()` function. It provides a nonlinear classification problem for comparing linear, polynomial, and RBF SVM kernels.

## Implementation

### Part A: Decision Trees

A Decision Tree classifier is implemented from scratch using Python and NumPy. The implementation includes:

- Gini impurity and entropy
- Finding the best feature and threshold for a split
- Recursive tree construction
- Stopping conditions and leaf predictions
- Prediction and printing decision rules

The manual implementation is compared with Scikit-learn's `DecisionTreeClassifier`. Decision boundaries are visualized, and experiments are conducted with different splitting criteria and maximum depths.

### Part B: Linear SVM

A linear SVM is implemented from scratch using gradient/subgradient updates based on the hinge-loss objective and a regularization term.

The implementation includes feature standardization, conversion of class labels to -1 and +1, prediction, and visualization of the decision boundary and margins.

Its performance is compared with Scikit-learn's linear SVM. Different values of the regularization parameter C, learning rate, and number of iterations are also tested.

### Part C: Nonlinear SVM

Scikit-learn SVM models are trained on the two-moons dataset using three kernels:

- **Linear:** Learns a straight decision boundary.
- **Polynomial:** Can learn more complex boundaries using polynomial relationships.
- **RBF:** Learns flexible nonlinear decision boundaries.

The models are compared using test accuracy and decision-boundary plots.

### Part D: Final Model Comparison

Five models are evaluated on the Banknote Authentication dataset using the same 70:30 training-test split:

1. Manual Decision Tree
2. Scikit-learn Decision Tree
3. Manual Linear SVM
4. Scikit-learn Linear SVM
5. RBF SVM

All four banknote features are used for this comparison. The SVM models use standardized features, with scaling parameters calculated from the training data.

The models are compared using accuracy, precision, recall, F1-score, training time, and prediction time.

## Results

### Decision Tree Experiments

Using variance and skewness, the manual Decision Tree matched Scikit-learn's predictions in the tested combinations of Gini/entropy and maximum depths 2, 3, 5, and 8.

The highest test accuracy among these experiments was approximately **93.20%**, achieved with the Gini criterion and maximum depth 5.

### Nonlinear SVM Experiments

On the two-moons dataset, the observed test accuracies were:

| Kernel | Test Accuracy |
|---|---:|
| RBF | 98.67% |
| Linear | 89.33% |
| Polynomial | 89.33% |

The RBF kernel performed best on this dataset because it was able to learn a curved decision boundary that followed the class structure more closely.

### Final Banknote Model Comparison

The observed test accuracies using all four features were:

| Model | Test Accuracy |
|---|---:|
| Manual Decision Tree | 98.30% |
| Scikit-learn Decision Tree | 98.54% |
| Manual Linear SVM | 97.33% |
| Scikit-learn Linear SVM | 98.06% |
| RBF SVM | 100.00% |

The RBF SVM achieved the highest accuracy on this particular test split. These results describe the selected split and do not guarantee the same performance on other unseen data.

The complete comparison table, including precision, recall, F1-score, and execution times, is generated in the notebook.

## Key Observations

- Decision Trees are relatively easy to interpret because their predictions can be explained through feature-based splitting rules.
- The manual Decision Tree matched Scikit-learn's predictions in the tested two-feature experiments.
- SVM performance depends on feature scaling and parameter selection.
- The RBF kernel performed better than the linear and polynomial kernels on the two-moons dataset in these experiments.
- Increasing model complexity does not always improve test performance.
- Accuracy alone does not describe every aspect of classification performance, so precision, recall, and F1-score are also considered.
- Training and prediction times help compare the computational cost of different implementations.

## Tools and Libraries

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Jupyter Notebook

## How to Run

1. Clone or download this repository.
2. Install the required libraries.
3. Open the assignment notebook in Jupyter Notebook or Visual Studio Code.
4. Run the cells in order, starting with the imports and dataset
