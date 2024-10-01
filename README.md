# Regularization-and-dimension-reduction
This project focuses on Regularization Techniques (Ridge and Lasso Regression) and Dimension Reduction (Principal Component Regression, or PCR) using the prostate dataset. Here's a detailed explanation of the steps involved in the project:
* Ridge Regression
The project begins by loading the prostate dataset and setting up the predictor variables (X) and response variable (y).
Ridge regression is performed using the glmnet package. The model is trained over a grid of lambda values to find the optimal regularization parameter.
The coefficients of the ridge regression model are examined, and the sum of the squared coefficients is plotted to visualize the effect of regularization.

* Cross-Validation for Ridge Regression
A 10-fold cross-validation (CV) approach is used to determine the optimal lambda for ridge regression. The model is fitted using half of the data, and the prediction error is computed on the other half.
The optimal lambda is chosen based on the minimum mean squared error (MSE) from the CV.

* Comparison with Ordinary Least Squares (OLS)
The predictions from OLS (using no regularization) are compared against those from the ridge regression model to evaluate the differences in prediction accuracy.

* Lasso Regression
Similar to ridge regression, Lasso regression (where 𝛼=1) is performed using the glmnet package.
A 10-fold cross-validation approach is again employed to select the optimal lambda for Lasso regression. The results are visualized.

* Dimension Reduction using Principal Component Regression (PCR)
The pls package is used to perform PCR on the prostate dataset.
The number of principal components is selected using both the "onesigma" and "randomization" methods, and validation plots are generated.

* Validation Set Approach for PCR
The data is split into a training set and a test set. PCR is performed on the training set, and the optimal number of components is selected using CV.
Predictions are made using the test set, and the MSE is computed to evaluate the model's performance.

* PCR via Eigenvalue Decomposition and SVD
Two alternative methods for implementing PCR are presented:
Eigenvalue Decomposition: The training data is standardized, and principal components are obtained using the correlation matrix.
Singular Value Decomposition (SVD): SVD is applied to the standardized training data, and predictions are made based on the first three principal components.
This project demonstrates the application of regularization techniques to reduce model complexity and prevent overfitting while exploring dimension reduction methods to improve predictive performance.

You can include this summary in your README file to provide an overview of the methods and findings in your project. Let me know if you need any further details or modifications!
