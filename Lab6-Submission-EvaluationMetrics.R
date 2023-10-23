

# STEP 1. Install and Load the Required Packages ----
## ggplot2 ----
if (require("ggplot2")) {
  require("ggplot2")
} else {
  install.packages("ggplot2", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## caret ----
if (require("caret")) {
  require("caret")
} else {
  install.packages("caret", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## mlbench ----
if (require("mlbench")) {
  require("mlbench")
} else {
  install.packages("mlbench", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## pROC ----
if (require("pROC")) {
  require("pROC")
} else {
  install.packages("pROC", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

## dplyr ----
if (require("dplyr")) {
  require("dplyr")
} else {
  install.packages("dplyr", dependencies = TRUE,
                   repos = "https://cloud.r-project.org")
}

# 1. Accuracy and Cohen's Kappa ----
# Accuracy is the percentage of correctly classified instances out of all
# instances. Accuracy is more useful in binary classification problems than
# in multi-class classification problems.

# On the other hand, Cohen's Kappa is similar to Accuracy, however, it is more
# useful for classification problems that do not have an equal distribution of
# instances amongst the classes in the dataset.

# For example, instead of Red are 50 instances and Blue are 50 instances,
# the distribution can be that Red are 70 instances and Blue are 30 instances.

## 1.a. Load the dataset ----
data(iris)
## 1.b. Determine the Baseline Accuracy ----
# Identify the number of instances that belong to each class (distribution or
# class breakdown).

# The result should show that 65% tested negative and 34% tested positive
# for diabetes.

# This means that an algorithm can achieve a 65% accuracy by
# predicting that all instances belong to the class "negative".

# This in turn implies that the baseline accuracy is 65%.
View(iris)
iris_freq <- iris$Species
cbind(frequency =
        table(iris_freq),
      percentage = prop.table(table(iris_freq)) * 100)

## 1.c. Split the dataset ----
# Define a 75:25 train:test data split of the dataset.
# That is, 75% of the original data will be used to train the model and
# 25% of the original data will be used to test the model.
train_index <- createDataPartition(iris$Species,
                                   p = 0.75,
                                   list = FALSE)
iris_train <- iris[train_index, ]
iris_test <- iris[-train_index, ]

## 1.d. Train the Model ----
# We apply the 5-fold cross validation resampling method
train_control <- trainControl(method = "cv", number = 5)

# We then train a Generalized Linear Model to predict the value of Diabetes
# (whether the patient will test positive/negative for diabetes).

# `set.seed()` is a function that is used to specify a starting point for the
# random number generator to a specific value. This ensures that every time you
# run the same code, you will get the same "random" numbers.
set.seed(7)
iris_model_rf  <-
  train(Species ~ ., data = iris_train, method = "rf",
        metric = "Accuracy", trControl = train_control)

## 1.e. Display the Model's Performance ----
### Option 1: Use the metric calculated by caret when training the model ----
# The results show an accuracy of approximately 77% (slightly above the baseline
# accuracy) and a Kappa of approximately 49%.
print(iris_model_rf)

### Option 2: Compute the metric yourself using the test dataset ----
# A confusion matrix is useful for multi-class classification problems.
# Please watch the following video first: https://youtu.be/Kdsp6soqA7o

# The Confusion Matrix is a type of matrix which is used to visualize the
# predicted values against the actual Values. The row headers in the
# confusion matrix represent predicted values and column headers are used to
# represent actual values.

predictions <- predict(iris_model_rf, newdata = iris_test[, 1:4])

# Make sure both 'predictions' and 'iris_test[, 1:4]$Species' are factors
predictions <- as.factor(predictions)
# Convert the "Species" column to a factor in the full data frame
iris_test$Species <- as.factor(iris_test$Species)

# Set factor levels for 'predictions' and 'iris_test[, 1:4]$Species'
confusion_matrix <- confusionMatrix(predictions, iris_test$Species)
print(confusion_matrix)

confusion_data <- as.data.frame(as.table(confusion_matrix))

### Option 3: Display a graphical confusion matrix ----
ggplot(data = confusion_data, aes(x = Reference, y = Prediction, fill = Freq)) +
  geom_tile() +
  geom_text(aes(label = Freq), vjust = 1) +
  scale_fill_gradient(low = "lightblue", high = "blue") +
  theme_minimal() +
  labs(title = "Confusion Matrix")


# 2. RMSE, R Squared, and MAE ----

# RMSE stands for "Root Mean Squared Error" and it is defined as the average
# deviation of the predictions from the observations.

# R Squared (R^2) is also known as the "coefficient of determination".
# It provides a goodness of fit measure for the predictions to the
# observations.

# NOTE: R Squared (R^2) is a value between 0 and 1 such that
# 0 refers to "no fit" and 1 refers to a "perfect fit".

## 2.a. Load the dataset ----
data(PimaIndiansDiabetes)

## 2.b. Split the dataset ----
# Define a train:test data split of the dataset. Such that 10/16 are in the
# train set and the remaining 6/16 observations are in the test set.

# In this case, we split randomly without using a predictor variable in the
# caret::createDataPartition function.

# For reproducibility; by ensuring that you end up with the same
# "random" samples

# We apply simple random sampling using the base::sample function to get
# 10 samples
set.seed(7)
train_index <- sample(1:nrow(PimaIndiansDiabetes), 10)
Pima_train <- PimaIndiansDiabetes[train_index, ]
Pima_test <- PimaIndiansDiabetes[-train_index, ]

## 2.c. Train the Model ----
# We apply bootstrapping with 1,000 repetitions
train_control <- trainControl(method = "boot", number = 1000)

# We then train a linear regression model to predict the value of Employed
# (the number of people that will be employed given the independent variables).
Pima_model_logistic <-
  train(diabetes ~ ., data = Pima_train,
        method = "glm", family = binomial,
        metric = "MSE",  
        trControl = train_control)

## 2.d. Display the Model's Performance ----
### Option 1: Use the metric calculated by caret when training the model ----
# The results show an RMSE value of approximately 4.3898 and
# an R Squared value of approximately 0.8594
# (the closer the R squared value is to 1, the better the model).
print(Pima_model_logistic)
### Option 2: Compute the metric yourself using the test dataset ----
predictions <- predict(Pima_model_logistic, Pima_test)
# These are the 6 values for employment that the model has predicted:
print(predictions)

# Calculate the mean squared error (MSE)
mse <- mean((as.numeric(predictions) - as.numeric(Pima_test$diabetes))^2)
# Calculate RMSE by taking the square root of MSE
rmse <- sqrt(mse)
print(paste("RMSE (MSE) =", rmse))

#### MAE ----
# MAE measures the average absolute differences between the predicted and
# actual values in a dataset. MAE is useful for assessing how close the model's
# predictions are to the actual values.

# MAE is expressed in the same units as the target variable, making it easy to
# interpret. For example, if you are predicting the amount paid in rent,
# and the MAE is KES. 10,000, it means, on average, your model's predictions
# are off by about KES. 10,000.

# 3. Area Under ROC Curve ----
# Area Under Receiver Operating Characteristic Curve (AUROC) or simply
# "Area Under Curve (AUC)" or "ROC" represents a model's ability to
# discriminate between two classes.

# ROC is a value between 0.5 and 1 such that 0.5 refers to a model with a
# very poor prediction (essentially a random prediction; 50-50 accuracy)
# and an AUC of 1 refers to a model that predicts perfectly.

# ROC can be broken down into:
## Sensitivity ----
#         The number of instances from the first class (positive class)
#         that were actually predicted correctly. This is the true positive
#         rate, also known as the recall.
## Specificity ----
#         The number of instances from the second class (negative class)
#         that were actually predicted correctly. This is the true negative
#         rate.

## 3.a. Load the dataset ----
data(iris)
## 3.b. Determine the Baseline Accuracy ----
# The baseline accuracy is 65%.
iris_species_freq <- iris$Species
cbind(frequency = table(iris_species_freq), 
      percentage = prop.table(table(iris_species_freq)) * 100)


## 3.c. Split the dataset ----
# Define an 80:20 train:test data split of the dataset.
set.seed(7)
train_index <- createDataPartition(iris$Species,
                                   p = 0.8,
                                   list = FALSE)
iris_train <- iris[train_index, ]
iris_test <- iris[-train_index, ]

## 3.d. Train the Model ----
# We apply the 10-fold cross validation resampling method
train_control <- trainControl(method = "cv", number = 10, classProbs = TRUE)


# We then train a k Nearest Neighbours Model to predict the value of Diabetes
# (whether the patient will test positive/negative for diabetes).
iris_model_knn <- train(Species ~ ., data = iris_train, method = "knn",
                        metric = "Accuracy", trControl = train_control)


## 3.e. Display the Model's Performance ----
### Option 1: Use the metric calculated by caret when training the model ----
# The results show a ROC value of approximately 0.76 (the closer to 1,
# the higher the prediction accuracy) when the parameter k = 9
# (9 nearest neighbours).

print(iris_model_knn)

### Option 2: Compute the metric yourself using the test dataset ----
#### Sensitivity and Specificity ----
predictions <- predict(iris_model_knn, iris_test)
# These are the values for diabetes that the
# model has predicted:
print(predictions)
confusion_matrix <-
  caret::confusionMatrix(predictions,
                         iris_test$Species)
# We can see the sensitivity (≈ 0.86) and the specificity (≈ 0.60) below:
print(confusion_matrix)

#### AUC ----
# The type = "prob" argument specifies that you want to obtain class
# probabilities as the output of the prediction instead of class labels.
predictions <- predict(iris_model_knn, iris_test, type = "prob")


# These are the class probability values for diabetes that the
# model has predicted:
print(predictions)

# "Controls" and "Cases": In a binary classification problem, you typically
# have two classes, often referred to as "controls" and "cases."
# These classes represent the different outcomes you are trying to predict.
# For example, in a medical context, "controls" might represent patients without
# a disease, and "cases" might represent patients with the disease.

# Setting the Direction: The phrase "Setting direction: controls < cases"
# specifies how you define which class is considered the positive class (cases)
# and which is considered the negative class (controls) when calculating
# Compute the ROC curves for each class
roc_curves <- lapply(unique(iris$Species), function(class_name) {
  class_predictions <- as.numeric(predictions[class_name] > 0)
  class_roc <- roc(iris_test$Species == class_name, class_predictions)
  return(class_roc)
})

# Plot the ROC curves for each class
colors <- rainbow(length(roc_curves))
for (i in 1:length(roc_curves)) {
  plot(roc_curves[[i]], col = colors[i], print.auc = TRUE, print.auc.x = 0.6, print.auc.y = 0.6, lwd = 2.5, add = (i > 1))
}

legend("bottomright", legend = unique(iris$Species), col = colors, lwd = 2.5)

# 4. Logarithmic Loss (LogLoss) ----
# Logarithmic Loss (LogLoss) is an evaluation metric commonly used for
# assessing the performance of classification models, especially when the model
# provides probability estimates for each class.

# LogLoss measures how well the predicted probabilities align with the true
# binary outcomes.

# In *binary classification*, the LogLoss formula for a single observation is:
# LogLoss = −(y log(p) + (1 − y)log(1 − p))

# Where:
# [*] y is the true binary label (0 or 1).
# [*] p is the predicted probability of the positive class.

# The LogLoss formula computes the logarithm of the predicted probability for
# the true class (if y = 1) or the logarithm of the predicted probability for
# the negative class (if y = 0), and then sums the results.

# A lower LogLoss indicates better model performance, where perfect predictions
# result in a LogLoss of 0.

########################### ----
## 4.a. Load the dataset ----
data(PimaIndiansDiabetes)

## 4.b. Train the Model ----
# We apply the 5-fold repeated cross validation resampling method
# with 3 repeats
train_control <- trainControl(method = "repeatedcv", number = 5, repeats = 3,
                              classProbs = TRUE,
                              summaryFunction = mnLogLoss)
set.seed(7)

# This creates a CART model. One of the parameters used by a CART model is "cp".
# "cp" refers to the "complexity parameter". It is used to impose a penalty to
# the tree for having too many splits. The default value is 0.01.
pima_model_cart <- train(diabetes ~ ., data = PimaIndiansDiabetes, method = "rpart",
                         metric = "logLoss", trControl = train_control)

## 4.c. Display the Model's Performance ----
### Option 1: Use the metric calculated by caret when training the model ----
# The results show that a cp value of ≈ 0 resulted in the lowest
# LogLoss value. The lowest logLoss value is ≈ 0.46.
print(pima_model_cart)

# [OPTIONAL] **Deinitialization: Create a snapshot of the R environment ----
# Lastly, as a follow-up to the initialization step, record the packages
# installed and their sources in the lockfile so that other team-members can
# use renv::restore() to re-install the same package version in their local
# machine during their initialization step.
# renv::snapshot() # nolint

# References ----

## Kuhn, M., Wing, J., Weston, S., Williams, A., Keefer, C., Engelhardt, A., Cooper, T., Mayer, Z., Kenkel, B., R Core Team, Benesty, M., Lescarbeau, R., Ziem, A., Scrucca, L., Tang, Y., Candan, C., & Hunt, T. (2023). caret: Classification and Regression Training (6.0-94) [Computer software]. https://cran.r-project.org/package=caret # nolint ----

## Leisch, F., & Dimitriadou, E. (2023). mlbench: Machine Learning Benchmark Problems (2.1-3.1) [Computer software]. https://cran.r-project.org/web/packages/mlbench/index.html # nolint ----

## National Institute of Diabetes and Digestive and Kidney Diseases. (1999). Pima Indians Diabetes Dataset [Dataset]. UCI Machine Learning Repository. https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database # nolint ----

## Robin, X., Turck, N., Hainard, A., Tiberti, N., Lisacek, F., Sanchez, J.-C., Müller, M., Siegert, S., Doering, M., & Billings, Z. (2023). pROC: Display and Analyze ROC Curves (1.18.4) [Computer software]. https://cran.r-project.org/web/packages/pROC/index.html # nolint ----

## Wickham, H., François, R., Henry, L., Müller, K., Vaughan, D., Software, P., & PBC. (2023). dplyr: A Grammar of Data Manipulation (1.1.3) [Computer software]. https://cran.r-project.org/package=dplyr # nolint ----

## Wickham, H., Chang, W., Henry, L., Pedersen, T. L., Takahashi, K., Wilke, C., Woo, K., Yutani, H., Dunnington, D., Posit, & PBC. (2023). ggplot2: Create Elegant Data Visualisations Using the Grammar of Graphics (3.4.3) [Computer software]. https://cran.r-project.org/package=ggplot2 # nolint ----

# **Required Lab Work Submission** ----
## Part A ----
# Create a new file called
# "Lab6-Submission-EvaluationMetrics.R".
# Provide all the code you have used to demonstrate the classification and
# regression evaluation metrics we have gone through in this lab.
# This should be done on any datasets of your choice except the ones used in
# this lab.

## Part B ----
# Upload *the link* to your
# "Lab6-Submission-EvaluationMetrics.R" hosted
# on Github (do not upload the .R file itself) through the submission link
# provided on eLearning.

## Part C ----
# Create a markdown file called "Lab-Submission-Markdown.Rmd"
# and place it inside the folder called "markdown". Use R Studio to ensure the
# .Rmd file is based on the "GitHub Document (Markdown)" template when it is
# being created.

# Refer to the following file in Lab 1 for an example of a .Rmd file based on
# the "GitHub Document (Markdown)" template:
#     https://github.com/course-files/BBT4206-R-Lab1of15-LoadingDatasets/blob/main/markdown/BIProject-Template.Rmd # nolint

# Include Line 1 to 14 of BIProject-Template.Rmd in your .Rmd file to make it
# displayable on GitHub when rendered into its .md version

# It should have code chunks that explain all the steps performed on the
# datasets.

## Part D ----
# Render the .Rmd (R markdown) file into its .md (markdown) version by using
# knitR in RStudio.

# You need to download and install "pandoc" to render the R markdown.
# Pandoc is a file converter that can be used to convert the following files:
#   https://pandoc.org/diagram.svgz?v=20230831075849

# Documentation:
#   https://pandoc.org/installing.html and
#   https://github.com/REditorSupport/vscode-R/wiki/R-Markdown

# By default, Rmd files are open as Markdown documents. To enable R Markdown
# features, you need to associate *.Rmd files with rmd language.
# Add an entry Item "*.Rmd" and Value "rmd" in the VS Code settings,
# "File Association" option.

# Documentation of knitR: https://www.rdocumentation.org/packages/knitr/

# Upload *the link* to "Lab-Submission-Markdown.md" (not .Rmd)
# markdown file hosted on Github (do not upload the .Rmd or .md markdown files)
# through the submission link provided on eLearning.