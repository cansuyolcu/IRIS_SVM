# IRIS_SVM
I will be analyzing the famous iris data set using support vector machines.
[The Iris flower data set](https://en.wikipedia.org/wiki/Iris_flower_data_set)  or Fisher's Iris data set is a multivariate data set introduced by Sir Ronald Fisher in the 1936 as an example of discriminant analysis.
The data set consists of 50 samples from each of three species of Iris (Iris setosa, Iris virginica and Iris versicolor), so 150 total samples. Four features were measured from each sample: the length and the width of the sepals and petals, in centimeters.

The iris dataset contains measurements for 150 iris flowers from three different species.

The three classes in the Iris dataset:

Iris-setosa (n=50)
Iris-versicolor (n=50)
Iris-virginica (n=50)
The four features of the Iris dataset:

sepal length in cm
sepal width in cm
petal length in cm
petal width in cm

# Getting the data

```python
import seaborn as sns
iris = sns.load_dataset('iris')
```

# Exploratory Data Analysis

```python
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline

iris.info()
```
<img src= "https://user-images.githubusercontent.com/66487971/88687666-3904a680-d101-11ea-8e84-46e16064685e.png" width = 350>

```python

sns.pairplot(iris,hue='species',palette = 'Dark2')
```

<img src= "https://user-images.githubusercontent.com/66487971/88687876-779a6100-d101-11ea-93cc-cf1b08af0570.png" width = 900>

It looks like Setosa is the most separable.

```python
setosa = iris[iris['species']== 'setosa']

sns.kdeplot(setosa['sepal_width'], setosa['sepal_length'],
                 cmap="plasma", shade=True, shade_lowest=False)
```

<img src= "https://user-images.githubusercontent.com/66487971/88688249-cb0caf00-d101-11ea-9f88-d0bb2521d325.png" width = 500>

## Train Test Split

```python
from sklearn.model_selection import train_test_split
X= iris.drop('species',axis=1)
y=iris['species']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
```
# Model Evaluation

```python
predictions = svc_model.predict(X_test)
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,predictions))
```

<img src= "https://user-images.githubusercontent.com/66487971/88688774-60a83e80-d102-11ea-9d2c-18be2820b309.png" width = 150>


```python

print(classification_report(y_test,predictions))

```

<img src= "https://user-images.githubusercontent.com/66487971/88688877-7c134980-d102-11ea-8319-92a98f94ea9c.png" width = 500>

It looks like it predicted all of them correctly.But I want to check it with GridSearch too.

```python
from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.1,1, 10, 100], 'gamma': [1,0.1,0.01,0.001]} 
grid = GridSearchCV(SVC(),param_grid,refit=True,verbose=2)
grid.fit(X_train,y_train)
```

<img src= "https://user-images.githubusercontent.com/66487971/88689249-f04ded00-d102-11ea-9b7b-064d1129868f.png" width = 1500>

```python
grid_predictions = grid.predict(X_test)
print(confusion_matrix(y_test,grid_predictions))

```

<img src= "https://user-images.githubusercontent.com/66487971/88689476-2f7c3e00-d103-11ea-825d-cdc7d1ce6b2b.png" width = 150>

```python
print(classification_report(y_test,grid_predictions))
```

<img src= "https://user-images.githubusercontent.com/66487971/88688877-7c134980-d102-11ea-8319-92a98f94ea9c.png" width = 500>

I still got 100% accuracy. I do not believe I overfit it anywhere so this was a success.

## This concludes my project here. Thanks for reading all the way through.






