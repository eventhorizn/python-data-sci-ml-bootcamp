# Machine Learning with Python

SciKit is the most popular python machine learning package

```
pip install scikit-learn
```

1. Every algorithm is exposed via an 'Estimator'
1. First you'll import the model
    ```py
    from sklearn.family import Model

    from skylearn.linear_model import LinearRegression
    ```
1. All parameters of an estimator can be set when instantiated
    ```py
    model = LinearRegresssion(normalize=True)
    print(model)

    LinearRegression(copy_X=True, fit_intercept=True, normalize=True)
    ```