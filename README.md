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

<img src= "https://user-images.githubusercontent.com/66487971/88688249-cb0caf00-d101-11ea-9f88-d0bb2521d325.png" width = 700>

## Train Test Split

