# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required libraries.
2. Upload and read the dataset.
3. Check for any null values using the isnull() function.
4. From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5. Find the accuracy of the model and predict the required values by importing the required module 
   from sklearn.

# Developed by:   VIDHIYA LAKSHMI S 
# RegisterNumber: 212223230238
## Program:
```
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by:   VIDHIYA LAKSHMI S 
RegisterNumber: 212223230238
import pandas as pd
data=pd.read_csv('/content/Employee.csv')
data.head()
data.info()
```
![image](https://github.com/user-attachments/assets/023c123f-efe0-4b1a-b5cf-05c71974e0fa)
```
data.isnull().sum()
data["left"].value_counts()
```
![image](https://github.com/user-attachments/assets/2ce121b7-c4b7-4db0-8ed7-ae091a494534)
```
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
```
![image](https://github.com/user-attachments/assets/71c5b36c-0005-4da0-ace6-023cdaba6cde)
```
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
```
![image](https://github.com/user-attachments/assets/970857a3-8736-4a0a-92ec-1858d0127b67)
```
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
```
```
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```
![image](https://github.com/user-attachments/assets/6855068f-9400-4b88-91cf-5b52caec3286)
```
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```
![image](https://github.com/user-attachments/assets/438e753e-f083-4fa0-ab0d-b81cc6d7d191)

## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
