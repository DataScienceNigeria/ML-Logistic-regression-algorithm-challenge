<h1 style='text-align: center'>HOW TO USE</h1>

<h3> Import Required external libraries</h3>


```python
import numpy as np
```

<h3> Import algorithm</h3>


```python
from custom_logistic_regression import CustomLogisticRegression as Lr
```

<h3>load dataset</h3>


```python
from sklearn.datasets import load_digits
x,y = load_digits(return_X_y = True)
```

<h3>Initialize model</h3>


```python
model = Lr()
```

<h3>Train Model</h3>


```python
model.fit(x,y)
```

<h3>Check accuracy of model</h3>


```python
model.accuracy()
```




    98.10795770728993



<h3>Make a prediction</h3>


```python
model.predict(x[500]),y[500]
```




    (8, 8)


