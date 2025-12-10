%let pgm=utl-altair-slc-imputing-missing-variables-subproces-k-nearest-neighbors-knn;

%stop_submission;

Altair slc imputing missing variables subproces-k-nn

Not sure this helps. If you provide sample input data I can rerun?.
I am a little out of my comfort zone.

Two Solutions Python and R

K-Nearest Neighbors (kNN) is a non-parametric supervised machine learning
algorithm used for classification and regression tasks.

community.altair
https://community.altair.com/discussion/65245/imputing-missing-variables-subprocess-k-nn?tab=all

/*                   _
(_)_ __  _ __  _   _| |_
| | `_ \| `_ \| | | | __|
| | | | | |_) | |_| | |_
|_|_| |_| .__/ \__,_|\__|
        |_|
*/

libname sd1 sas7bdat "d:/sd1";
options validvarname=v7;
data sd1.have;
input xxa xxb yya;
cards4;
1.0 2.0 0
1.5 1.8 0
2.0 2.2 0
2.5 2.0 1
3.0 3.5 1
3.5 3.0 1
;;;;
run;quit;

/*               _   _
/ |  _ __  _   _| |_| |__   ___  _ __
| | | `_ \| | | | __| `_ \ / _ \| `_ \
| | | |_) | |_| | |_| | | | (_) | | | |
|_| | .__/ \__, |\__|_| |_|\___/|_| |_|
    |_|    |___/
*/

%utl_slc_pybeginx(
   return=date                    /*-  return date            -*/
  ,resolve=Y                      /*- resolve macros in python-*/
  ,in=d:/sd1/have.sas7bdat        /*- input sas dataset       -*/
  ,out=cm_df_with_totals          /*- output work.female      -*/
  ,py2r=c:/temp/py_dataframe.rds  /*- py 2 r dataframe        -*/
  );
cards4;
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pyreadstat as ps
import pyreadr as pr

df,meta = ps.read_sas7bdat("&in")
print(df)

# Assuming your DataFrame is called 'df'
X = df[['xxa', 'xxb']].to_numpy().astype(float)
y = df['yya'].to_numpy()
print(X)
print(y)

# Introduce missing values
X[1, 0] = np.nan   # feature 1 of sample 2
X[4, 1] = np.nan   # feature 2 of sample 5

# Train / test split (small just for illustration)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.33, random_state=42, stratify=y
)

# Pipeline: KNN imputation -> KNN classifier
pipe = Pipeline([
    ("imputer", KNNImputer(n_neighbors=2)),
    ("knn", KNeighborsClassifier(n_neighbors=3))
])

pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)

print("X_train with missing values:\n", X_train)
print("X_test with missing values:\n", X_test)
print("Predicted labels:", y_pred)
print("True labels     :", y_test)
print("Accuracy        :", accuracy_score(y_test, y_pred))

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Create a pandas DataFrame for the confusion matrix
# Option 1: Simple DataFrame with row/column labels
cm_df = pd.DataFrame(
    cm,
    index=[f'Actual Class {i}' for i in range(cm.shape[0])],
    columns=[f'Predicted Class {i}' for i in range(cm.shape[1])]
)

print("\nConfusion Matrix as DataFrame:")
print(cm_df)

# Option 2: More detailed DataFrame with proper labeling
# Get unique class labels from y_test
classes = np.unique(np.concatenate([y_test, y_pred]))
cm_df_detailed = pd.DataFrame(
    cm,
    index=[f'Actual: {cls}' for cls in classes],
    columns=[f'Predicted: {cls}' for cls in classes]
)

print("\nDetailed Confusion Matrix DataFrame:")
print(cm_df_detailed)

# Option 3: With additional metrics
# Add row and column sums
cm_df_with_totals = pd.DataFrame(cm)
cm_df_with_totals['Actual Total'] = cm_df_with_totals.sum(axis=1)
cm_df_with_totals.loc['Predicted Total'] = cm_df_with_totals.sum(axis=0)
cm_df_with_totals = cm_df_with_totals.rename(
    index={i: f'Actual Class {i}' for i in range(len(classes))},
    columns={i: f'Predicted Class {i}' for i in range(len(classes))}
)

print("\nConfusion Matrix with Totals:")
print(cm_df_with_totals)

# You can also get a classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=[f'Class {i}' for i in classes]))

pr.write_rds("&py2r",&out) # panda datafame 2 r dataframe
;;;;
%utl_slc_pyendx;

proc print data=cm_df_with_totals;
run;quit;

/*           _         _
  ___  _   _| |_ _ __ | |_
 / _ \| | | | __| `_ \| __|
| (_) | |_| | |_| |_) | |_
 \___/ \__,_|\__| .__/ \__|
                |_|
*/

X_train with missing values:
 [[nan 1.8]
 [3.5 3. ]
 [1.  2. ]
 [3.  nan]]

X_test with missing values:
 [[2.5 2. ]
 [2.  2.2]]

Predicted labels: [1. 0.]
True labels     : [1. 0.]
Accuracy        : 1.0

Confusion Matrix:
[[1 0]
 [0 1]]

Confusion Matrix as DataFrame:
                Predicted Class 0  Predicted Class 1
Actual Class 0                  1                  0
Actual Class 1                  0                  1

Detailed Confusion Matrix DataFrame:
             Predicted: 0.0  Predicted: 1.0
Actual: 0.0               1               0
Actual: 1.0               0               1

Confusion Matrix with Totals:
                 Predicted Class 0  Predicted Class 1  Actual Total
Actual Class 0                   1                  0             1
Actual Class 1                   0                  1             1
Predicted Total                  1                  1             2

Classification Report:
              precision    recall  f1-score   support

   Class 0.0       1.00      1.00      1.00         1
   Class 1.0       1.00      1.00      1.00         1

    accuracy                           1.00         2
   macro avg       1.00      1.00      1.00         2
weighted avg       1.00      1.00      1.00         2


Altair SLC (SLC Table - SAS dataset (confusion matrix back to slc perfect classification with imputed data)

  Predicted Class 0 Predicted Class 1 Actual Total
1                 1                 0            1
2                 0                 1            1
3                 1                 1            2

/*___
|___ \   _ __
  __) | | `__|
 / __/  | |
|_____| |_|

*/

libname sd1 sas7bdat "d:/sd1";
options validvarname=v7;
data sd1.have;
input x1 x2 y;
cards4;
1.0 2.0 0
1.5 1.8 0
2.0 2.2 0
2.5 2.0 1
3.0 3.5 1
3.5 3.0 1
;;;;
run;quit;


options set=RHOME "D:\d451";
proc r;
export data=sd1.have r=dat;
submit;
set.seed(123)
dat

dat$y = as.factor(dat$y)

## Introduce missing values
dat$x1[2] <- NA   # x1 of 2nd row
dat$x2[5] <- NA   # x2 of 5th row

## Train / test split (indices chosen for reproducibility)
train_idx <- c(1, 2, 4, 5)
test_idx  <- c(3, 6)

train <- dat[train_idx, ]
test  <- dat[test_idx, ]

## ---- Simple mean imputation function ----
mean_impute <- function(train, test, cols) {
  train_imp <- train
  test_imp  <- test

  for (j in cols) {
    m <- mean(train_imp[[j]], na.rm = TRUE)
    train_imp[[j]][is.na(train_imp[[j]])] <- m
    test_imp[[j]][is.na(test_imp[[j]])]   <- m
  }

  list(train = train_imp, test = test_imp)
}

num_cols <- c("x1", "x2")
imp <- mean_impute(train, test, num_cols)

train_imp <- imp$train
test_imp  <- imp$test

## ---- Plain k-NN implementation (Euclidean) ----
knn_manual <- function(train_x, train_y, test_x, k = 3) {
  n_test <- nrow(test_x)
  preds  <- character(n_test)

  for (i in seq_len(n_test)) {
    # compute distances from test point i to all train points
    d <- sqrt(rowSums((t(t(train_x) - as.numeric(test_x[i, ])))^2))

    # indices of k nearest neighbors
    nn_idx <- order(d)[1:k]

    # majority vote
    nn_classes <- train_y[nn_idx]
    tab <- table(nn_classes)
    preds[i] <- names(tab)[which.max(tab)]
  }

  factor(preds, levels = levels(train_y))
}

train_x <- train_imp[, num_cols]
train_y <- train_imp$y

test_x  <- test_imp[, num_cols]
test_y  <- test_imp$y

pred_y <- knn_manual(train_x, train_y, test_x, k = 3)

confusion<-data.frame(test_x, True = test_y, Pred = pred_y)

## Confusion matrix
print(confusion)

print(table(True = test_y, Pred = pred_y))
endsubmit;
import data=confusion  r=confusion;

run;quit;

proc print data=confusion;
run;quit;

/*           _               _
  ___  _   _| |_ _ __  _   _| |_
 / _ \| | | | __| `_ \| | | | __|
| (_) | |_| | |_| |_) | |_| | |_
 \___/ \__,_|\__| .__/ \__,_|\__|
                |_|
*/

Altair SLC

   x1  x2 y
1 1.0 2.0 0
2 1.5 1.8 0
3 2.0 2.2 0
4 2.5 2.0 1
5 3.0 3.5 1
6 3.5 3.0 1

   x1  x2 True Pred
3 2.0 2.2    0    0
6 3.5 3.0    1    1

    Pred
True 0 1
   0 1 0
   1 0 1

Altair SLC (confusion matrix back to slc perfect classification with imputed data)

Obs    x1     x2     True    Pred

 1     2.0    2.2     0       0
 2     3.5    3.0     1       1

/*              _
  ___ _ __   __| |
 / _ \ `_ \ / _` |
|  __/ | | | (_| |
 \___|_| |_|\__,_|

*/
