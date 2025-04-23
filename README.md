# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:

To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:

1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm

STEP 1. Start the program.

STEP 2. Data Preprocessing: Read dataset, drop unnecessary columns, and encode categorical variables.

STEP 3. Initialize Parameters: Initialize theta randomly and extract features (x) and target variable (y).

STEP 4. Define Sigmoid Function: Implement the sigmoid function to transform linear outputs into probabilities.

STEP 5. Define Loss Function and Gradient Descent: Define loss function using sigmoid output and implement gradient descent to minimize loss.

STEP 6. Prediction and Evaluation: Use trained parameters to predict on dataset, calculate accuracy, and optionally predict placement status of new data points.

STEP 7. End the program.
<br>
<br>
<br>
<br>
<br>
<br>
## Program:

```python
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by:   S JAIGANESH
RegisterNumber:  212222240037
```
```py
import pandas as pd
import numpy as np

data=pd.read_csv('/content/Placement_Data.csv')
```
```py
data=data.drop('sl_no',axis=1)
data=data.drop('salary',axis=1)
```
```py
data["gender"]=data["gender"].astype('category')
data["ssc_b"]=data["ssc_b"].astype('category')
data["hsc_b"]=data["hsc_b"].astype('category')
data["degree_t"]=data["degree_t"].astype('category')
data["workex"]=data["workex"].astype('category')
data["specialisation"]=data["specialisation"].astype('category')
data["status"]=data["status"].astype('category')
data["hsc_s"]=data["hsc_s"].astype('category')
data.dtypes
```
```py
data["gender"]=data["gender"].cat.codes
data["ssc_b"]=data["ssc_b"].cat.codes
data["hsc_b"]=data["hsc_b"].cat.codes
data["degree_t"]=data["degree_t"].cat.codes
data["workex"]=data["workex"].cat.codes
data["specialisation"]=data["specialisation"].cat.codes
data["status"]=data["status"].cat.codes
data["hsc_s"]=data["hsc_s"].cat.codes
data
```
```py
x=data.iloc[:,:-1].values
y=data.iloc[:,-1].values
y
```
```py
theta = np.random.randn(x.shape[1])
Y=y
```
```py
def sigmoid(z):
  return 1/(1+np.exp(-z))
```
```py
def loss(theta,X,y):
  h=sigmoid(X.dot(theta))
  return -np.sum(y*np.log(h)+(1-y)*np.log(1-h))
```
```py
def gradient_descent(theta,X,y,alpha,num_iterations):
  m=len(y)
  for i in range(num_iterations):
    h=sigmoid(X.dot(theta))
    gradient = X.T.dot(h-y)/m
    theta-=alpha * gradient
  return theta
```
```py
theta=gradient_descent(theta,x,y,alpha=0.01,num_iterations=1000)
```
```py
def predict(theta,X):
  h=sigmoid(X.dot(theta))
  y_pred=np.where(h>=0.5,1,0)
  return y_pred 

y_pred = predict(theta,x)
```
```py
y_pred=predict(theta,x)
```
```py
accuracy=np.mean(y_pred.flatten()==y)
print("Accuracy: ",accuracy)
```
```py
print(y_pred)
```
```py
xnew=np.array([[0,87,0,95,0,2,78,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)
```
```py
xnew=np.array([[0,0,0,0,0,2,8,2,0,0,1,0]])
y_prednew=predict(theta,xnew)
print(y_prednew)
```
## Output:

### data columns

![image](https://github.com/SanjayRagavendar/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/91368803/3aec9c46-c885-42bb-8b45-215ac5d6274f)

### data after encoding

![image](https://github.com/SanjayRagavendar/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/91368803/b6a86619-04fe-4560-a310-194c80aa728a)

<br>
<br>
<br>
<br>
<br>
<br> 

### Array value of Y:

![image](https://github.com/SanjayRagavendar/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/91368803/a7b6a576-f916-4792-a189-9df2e8ca530f)



### Accuracy

![image](https://github.com/SanjayRagavendar/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/91368803/12a41cc4-18b2-4375-a7eb-9d1c4a554cc5)


  
### New accuracy
![image](https://github.com/SanjayRagavendar/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/91368803/36fd6f69-d021-4489-b46a-ab4f68f410ea)

### New accuracy
![image](https://github.com/SanjayRagavendar/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/91368803/36fd6f69-d021-4489-b46a-ab4f68f410ea)

## Result:

Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.
