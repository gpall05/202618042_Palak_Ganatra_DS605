# PCA and Face Recognition — DS605 Lab 7

## About the Project

This project explores how face images can be represented using numerical features and how Principal Component Analysis (PCA) can reduce the number of features without losing too much information. The reduced representations are then used to reconstruct images and recognize faces.

The work is based on the AT&T (ORL) face dataset and was completed as part of the DS605 Fundamentals of Machine Learning lab assignment.

## Dataset

The dataset contains 400 grayscale face images of 40 subjects, with 10 images per subject. Each image has a resolution of 112 × 92 pixels, giving 10,304 pixel values per image.

The images were converted into one-dimensional feature vectors and combined into a feature matrix. Subject identities were stored separately as labels. The data was divided into training and testing sets using an 80:20 stratified split, with a fixed random seed to make the results reproducible.

## Approach

### 1. Image Representation

Each grayscale image was flattened into a vector of 10,304 pixel values. All vectors were combined into a single feature matrix, with the corresponding subject identities stored as labels.

### 2. PCA and Dimensionality Reduction

The mean face was calculated using the training data, and the training and test images were centered using this mean. PCA was implemented using NumPy's linear algebra functions rather than Scikit-learn's PCA implementation, as required by the assignment.

Both Singular Value Decomposition (SVD) and eigen-decomposition were used to obtain the principal directions. Their leading eigenvalues and directions were compared to check that the two approaches agreed.

The cumulative explained variance was then used to determine how many components were needed to retain at least 90% of the training variance.

### 3. Eigenfaces and Reconstruction

The principal directions were reshaped into images to visualize eigenfaces. Test images were also reconstructed from their reduced PCA representations. The reconstruction quality was examined visually and evaluated using mean squared error (MSE).

### 4. Face Recognition

The reduced representations were used to identify test faces by finding the nearest training image in PCA space using Euclidean distance.

K-Nearest Neighbors (KNN), Support Vector Machine (SVM) with an RBF kernel, and Logistic Regression were also trained using the PCA features. Their test accuracies were compared with the nearest-neighbor baseline.

Finally, different numbers of principal components were tested to examine the effect of representation size on recognition accuracy.

## Results and Observations

Using 98 principal components retained approximately 90.06% of the training variance. This reduced the number of features per image from 10,304 to 98, a reduction of approximately 99.05%.

The mean squared reconstruction error on the test images was 329.983.

The initial classifier comparison produced the following results:

| Method | Test Accuracy |
|---|---:|
| PCA Nearest Neighbor | 95.00% |
| KNN (3 neighbors) | 91.25% |
| SVM (RBF kernel) | 96.25% |
| Logistic Regression | 95.00% |

In the component-count experiment, SVM achieved 97.50% accuracy with both 50 and 250 components. The 50-component representation retained 82.48% of the training variance, whereas 250 components retained 98.60%.

This experiment showed that retaining more variance did not necessarily improve recognition accuracy. A smaller representation performed equally well for SVM on this particular test split.

## Conclusion

The experiments demonstrated that PCA can represent face images using far fewer features while retaining information useful for face recognition. The comparison between SVD and eigen-decomposition also showed that the two methods produced matching leading principal directions.

Among the initial classifier configurations, SVM achieved the highest test accuracy. The component-count experiment further showed that 50 components were sufficient to achieve the best observed SVM accuracy in this experiment.

These results are based on one train-test split, so the best component count and recognition accuracy may differ on other splits. Explained variance measures how much variation is retained in the data, while classification accuracy measures how well the models identify subjects; the two measures are not equivalent.

## Requirements

The notebook uses Python and the following libraries:

- NumPy
- Matplotlib
- Pillow
- Scikit-learn
- Pandas

## How to Run

1. Clone or download this repository.
2. Install the required libraries if they are not already available.
3. Place the AT&T/ORL dataset in the expected folder structure, with subject folders such as `s1`, `s2`, and so on, or use the provided ZIP file if included.
4. Open the notebook in Jupyter Notebook or VS Code.
5. Run the cells in order from beginning to end.

The notebook contains the implementation, visualizations, reconstruction examples, recognition results, and classifier comparisons.

## Implementation Note

PCA is implemented using NumPy linear algebra, as required by the assignment. Scikit-learn is used for the final traditional machine-learning classifiers and the train-test split, not for PCA itself.
