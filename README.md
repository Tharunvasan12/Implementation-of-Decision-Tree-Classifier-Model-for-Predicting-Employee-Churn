# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import pandas
2. Import Decision tree classifier
3. Fit the data in the model
4. Find the accuracy score

## Program:

### Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

__Developed by: THARUNISH VASAN T__

__Register Number: 212224240174__

```py
import pandas as pd
data=pd.read_csv("Employee.csv")
print("data.head():")
data.head()
```

```py
print("data.info():")
data.info()
```

```py
print("isnull() and sum():")
data.isnull().sum()
```

```py
print("data value counts():")
data["left"].value_counts()
```

```py
print("data.head() for Salary:")
data["salary"]=le.fit_transform(data["salary"])
data.head()
```

```py
print("x.head():")
x=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
```

```py
y=data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt=DecisionTreeClassifier(criterion="entropy")
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)
```

```py
print("Accuracy value:")
from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

```py
print("Data Prediction:")
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

```py
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

plt.figure(figsize=(8,6))
plot_tree(dt, feature_names=x.columns, class_names=['salary', 'left'], filled=True)
plt.show()
```
## Output:

<img width="1314" height="221" alt="image" src="https://github.com/user-attachments/assets/f79526be-2914-460c-99a2-e2e1dce671b6" />


<img width="852" height="312" alt="image" src="https://github.com/user-attachments/assets/297aeab6-8935-4f3e-92c5-67044eeedbbb" />


<img width="814" height="401" alt="image" src="https://github.com/user-attachments/assets/b1c96a14-8a7b-46f2-b515-384f1ab603d9" />


<img width="692" height="189" alt="image" src="https://github.com/user-attachments/assets/149d6581-45dc-42ba-af55-a919cfc08247" />


<img width="1316" height="219" alt="image" src="https://github.com/user-attachments/assets/1b55a02b-58a3-45e9-be5f-aa03a7aed004" />


<img width="1198" height="211" alt="image" src="https://github.com/user-attachments/assets/6450bc76-6f42-4d01-a0a0-acb8b5a68e4b" />



<img width="795" height="44" alt="image" src="https://github.com/user-attachments/assets/a077a69f-8cb1-4a0a-9f3b-babda968584a" />


<img width="1426" height="74" alt="image" src="https://github.com/user-attachments/assets/ba02fbfb-363b-4d11-a51a-11b3882728d5" />



<img width="653" height="482" alt="image" src="https://github.com/user-attachments/assets/93154508-c20b-4e7a-b95a-7fbde2c4858b" />



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
