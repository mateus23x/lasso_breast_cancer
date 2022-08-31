[github.com/mateus23x/lasso_breast_cancer](https://github.com/mateus23x/lasso_breast_cancer)

May 7, 2022




# What is Normalization?

The normalization $y$ of a vector $v$ is given by

$$y = \dfrac{v}{∥ v ∥}$$

In L1 normalizations:

$$ ∥v∥_{1} = \sum\limits_{i=1}^{n} | v_{i} | $$


The norm one of this vector is:

$$ ∥y∥_{1} = 1 $$

And it has the same direction as the original vector $v$, meaning that $y$ is proportional to $v$.

## Lasso Regression

The lasso coefficients, $\hat{\beta} _{\lambda}^{L}$ , minimize the following cost function:

$$ \sum_{i=1}^{n} \left( y_{i} - \beta_{0} - \sum_{j=1}^{p} \beta_{j} x_{ij} \right) + \lambda \sum\limits_{j=1}^{p} | \beta_{j} | $$

Where:

$n$ - lenght of the data

$p$ - number of $\beta^{L}$ coeficients

$\lambda \ge 0$ is a tuning parameter to increase the effect of the regularization L1 performed by $\sum\limits_{j=1}^{p} | \beta_{j} |$

Also:

$\lambda$ affects the speed and "quality" of the learning process.


So, the Lasso Regression seeks coeficients $\hat{\beta} _{i}^{L}$ for all $i \in \mathbb{N}$ that makes the Residual Sum of Squares smaller, but considering the minimum sum of module of all $\hat{\beta} _{i}$.

In this case, the penalty has the effect of forcing some of the coefficient estimates to be exactly equal to zero when the tuning parameter λ is sufficiently large. This results in an automatic feature selection, removing features that do not contribute much to the prediction task.

## Our problem

Breast cancer affects the lives of 2 million people annually in Brazil (source: Hospital Israelita A. Einstein). 

Early diagnosis can give better quality and save people's lives.

## Python Setup


```python
# built-in
import warnings

# third-party
import numpy as np
import pandas as pd
import sklearn.datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RepeatedKFold
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
```


```python
warnings.filterwarnings("ignore")
```

# The data

Loading data from sklearn datasets library:


```python
breast_cancer_dataset = sklearn.datasets.load_breast_cancer()
```


```python
# splitting vectors
X = pd.DataFrame(
    breast_cancer_dataset.data,
    columns=breast_cancer_dataset.feature_names
)
Y = pd.Series(breast_cancer_dataset.target, name="target")
```

We have 30 descriptive features.

Some will contribute more to the results of the model, others less or not at all.


```python
data = pd.concat([X, Y], axis=1)
print("Features: ", X.columns)
```

    Features:  Index(['mean radius', 'mean texture', 'mean perimeter', 'mean area',
           'mean smoothness', 'mean compactness', 'mean concavity',
           'mean concave points', 'mean symmetry', 'mean fractal dimension',
           'radius error', 'texture error', 'perimeter error', 'area error',
           'smoothness error', 'compactness error', 'concavity error',
           'concave points error', 'symmetry error', 'fractal dimension error',
           'worst radius', 'worst texture', 'worst perimeter', 'worst area',
           'worst smoothness', 'worst compactness', 'worst concavity',
           'worst concave points', 'worst symmetry', 'worst fractal dimension'],
          dtype='object')



```python
# splitting data
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=123)
```

# The model


```python
# defining the model and the evaluation method
lasso_model = LogisticRegression(penalty="l1", solver="liblinear", max_iter=1000)
cross_validation = RepeatedKFold(n_splits=10, n_repeats=3, random_state=321)
```

## Hyperparameter Tuning


```python
# defining the Grid Search Cross Validation and passing the search parameters
grid = {"C": np.arange(0.01, 1, 0.01)} # this parameter "C" is the ʎ
search = GridSearchCV(
    lasso_model,
    grid,
    scoring="neg_mean_absolute_error",
    cv=cross_validation,
    n_jobs=-1
)
```


```python
# tunning the Lasso Regression model hiperparameters with Grid Search Cross Validation
results = search.fit(x_train, y_train)
lasso_model.fit(x_train, y_train)
```



```python
print("MAE = %s" % round(results.best_score_, 5))
print("The optimum ʎ =", results.best_params_["C"])
```

    MAE = -0.05203
    The optimum ʎ = 0.88


## Feature selection

As we can see below, some of the coefficients are now zero.

Keeping these features increases generalism but reduces the model accuracy.


```python
feature_importance = dict(zip(data.columns, lasso_model.coef_[0]))
feature_importance
```




    {'mean radius': 4.72277103317473,
     'mean texture': 0.13910784187609024,
     'mean perimeter': -0.3214483071526125,
     'mean area': -0.015060792693121432,
     'mean smoothness': 0.0,
     'mean compactness': 0.0,
     'mean concavity': 0.0,
     'mean concave points': 0.0,
     'mean symmetry': 0.0,
     'mean fractal dimension': 0.0,
     'radius error': 0.0,
     'texture error': 1.7735895811251623,
     'perimeter error': 0.0,
     'area error': -0.09404347741989556,
     'smoothness error': 0.0,
     'compactness error': 0.0,
     'concavity error': 0.0,
     'concave points error': 0.0,
     'symmetry error': 0.0,
     'fractal dimension error': 0.0,
     'worst radius': 0.0,
     'worst texture': -0.39516249649609014,
     'worst perimeter': -0.0528365327779002,
     'worst area': -0.014988117206498891,
     'worst smoothness': 0.0,
     'worst compactness': 0.0,
     'worst concavity': -3.3574013387982156,
     'worst concave points': 0.0,
     'worst symmetry': 0.0,
     'worst fractal dimension': 0.0}




```python
selected_ = [k for k, v in feature_importance.items() if v != 0]
print("Selected features: ", selected_)
```

    Selected features:  [
        'mean radius',
        'mean texture',
        'mean perimeter',
        'mean area',
        'texture error',
        'area error',
        'worst texture',
        'worst perimeter',
        'worst area',
        'worst concavity'
    ]



```python
x_train_selected = x_train[selected_]
x_test_selected = x_test[selected_]
```

## Fitting the model with the best hyperparameter and the selected features


```python
lasso_model = LogisticRegression(
    C=results.best_params_["C"],
    penalty="l1",
    solver="liblinear",
    max_iter=1000
)
```


```python
lasso_model.fit(x_train_selected, y_train)
```



## Making predictions and checking the model score


```python
print("Accuracy on training data:", accuracy_score(lasso_model.predict(x_train_selected), y_train))
```

    Accuracy on training data: 0.9538461538461539



```python
accuracy_score(lasso_model.predict(x_test_selected), y_test)
```




    0.9824561403508771


There is no aparent overfitting right here.

And the model performed well on the test data.
