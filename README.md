# Implementation-of-Logistic-Regression-Using-Gradient-Descent

## AIM:
To write a program to implement the the Logistic Regression Using Gradient Descent.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the data file and import numpy, matplotlib and scipy.
2. Visulaize the data and define the sigmoid function, cost function and gradient descent.
3. Plot the decision boundary.
4. Calculate the y-prediction.

## Program:
```
/*
Program to implement the the Logistic Regression Using Gradient Descent.
Developed by: Kavya K
RegisterNumber:  212222230065
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize    #to remove unwanted data and memory storage

data=np.loadtxt("/content/ex2data1 (1).txt",delimiter=',')
X=data[:,[0,1]]
y=data[:,2]

X[:5]

y[:5]

Visualizing the data
plt.figure()
plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not admitted")
plt.xlabel("Exam 1 score")
plt.ylabel("Exam 2 score")
plt.legend()
plt.show()

Sigmoid fuction
def sigmoid(z):
  return 1/(1+np.exp(-z))
  
plt.plot()
X_plot=np.linspace(-10,10,100)
plt.plot(X_plot, sigmoid(X_plot))
plt.show()

def costFuction(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  J= -(np.dot(y, np.log(h)) + np.dot(1-y,np.log(1-h))) / X.shape[0]
  grad = np.dot(X.T, h-y) / X.shape[0]
  return J,grad
  
X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([0,0,0])
J, grad=costFuction(theta, X_train, y)
print(J)
print(grad)

X_train=np.hstack((np.ones((X.shape[0],1)),X))
theta=np.array([-24,0.2,0.2])
J, grad=costFuction(theta, X_train, y)
print(J)
print(grad)

def cost(theta,X,y):
  h = sigmoid(np.dot(X,theta))
  J= -(np.dot(y, np.log(h)) + np.dot(1-y, np.log(1-h))) / X.shape[0]
  return J
  
def gradient(theta,X,y):
  h=sigmoid(np.dot(X,theta))
  grad= np.dot(X.T, h-y) / X.shape[0]
  return grad
  
X_train = np.hstack((np.ones((X.shape[0],1)),X))
theta= np.array([0,0,0])
res = optimize.minimize(fun=cost, x0=theta, args=(X_train,y),method="Newton-CG",jac=gradient)
print(res.fun)
print(res.x)

def plotDecisionBoundary(theta,X,y):
  x_min, x_max = X[:,0].min() - 1, X[:,0].max()+1
  y_min, y_max = X[:,1].min() - 1, X[:,1].max()+1
  xx, yy = np.meshgrid(np.arange(x_min,x_max,0.1),
                       np.arange(y_min,y_max,0.1))
  X_plot = np.c_[xx.ravel(), yy.ravel()]
  X_plot = np.hstack((np.ones((X_plot.shape[0],1)),X_plot))
  y_plot = np.dot(X_plot, theta).reshape(xx.shape)

  plt.figure()
  plt.scatter(X[y==1][:,0],X[y==1][:,1],label="Admitted")
  plt.scatter(X[y==0][:,0],X[y==0][:,1],label="Not admitted")
  plt.contour(xx,yy,y_plot, levels=[0])
  plt.xlabel("Exam 1 score")
  plt.ylabel("Exam 2 score")
  plt.legend()
  plt.show()
  
  plotDecisionBoundary(res.x,X,y)
  
prob = sigmoid(np.dot(np.array([1,45,85]),res.x))
print(prob)

def predict(theta,X):
  X_train = np.hstack((np.ones((X.shape[0],1)),X))
  prob = sigmoid(np.dot(X_train,theta))
  return(prob >= 0.5).astype(int)
  
np.mean(predict(res.x,X)==y)
*/
```
## Output:
## Array Value of x:
![image](https://github.com/kavyasenthamarai/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118668727/d9fabec3-e193-4df5-93a7-9f6f0557d4dd)

## Array Value of y:
![image](https://github.com/kavyasenthamarai/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118668727/6bc5876e-aa56-4090-9362-31c50ac74f01)

## Exam 1 - score graph:
![image](https://github.com/kavyasenthamarai/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118668727/b4015f6a-3eb8-4118-b429-d38665954305)


## Sigmoid function graph:
![image](https://github.com/kavyasenthamarai/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118668727/ed30dc18-b7b3-48d5-815b-5ae343c78505)


## X_train_grad value:
![image](https://github.com/kavyasenthamarai/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118668727/2c643f4a-61dd-4aeb-aa56-87b43b4a2a78)


## Y_train_grad value:
![image](https://github.com/kavyasenthamarai/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118668727/0a990105-4ab1-4a0c-8eb2-3cdb32c58668)


## Print res.x:
![image](https://github.com/kavyasenthamarai/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118668727/6b5f8226-7ae2-4a72-9789-72b77fb481e6)


## Decision boundary - graph for exam score:
![image](https://github.com/kavyasenthamarai/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118668727/97cefd0a-2527-43e2-8677-f8ab3ce3bc3c)


## Proability value:
![image](https://github.com/kavyasenthamarai/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118668727/e776f1a7-d997-478b-922a-3776f3143a90)


## Prediction value of mean:
![image](https://github.com/kavyasenthamarai/-Implementation-of-Logistic-Regression-Using-Gradient-Descent/assets/118668727/c45d2190-572b-46c1-8793-534a5845d5dc)


## Result:
Thus the program to implement the the Logistic Regression Using Gradient Descent is written and verified using python programming.

