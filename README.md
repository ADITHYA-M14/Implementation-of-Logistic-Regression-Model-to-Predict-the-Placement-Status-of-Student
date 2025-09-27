# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data.
2. Print the placement data and salary data.
3. Find the null and duplicate values.
4. Using logistic regression find the predicted values of accuracy , confusion matrices.
5. 5. Display the results

## Program:
/*

Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

Developed by: ADITHYA M

RegisterNumber: 212224230008

*/

```
    import pandas as pd
    data=pd.read_csv("Placement_Data.csv")
    data.head()
    
    data1=data.copy()
    data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
    data1.head()
    
    data1.isnull().sum()
    
    data1.duplicated().sum()
    
    from sklearn.preprocessing import LabelEncoder
    le=LabelEncoder()
    data1["gender"]=le.fit_transform(data1["gender"])
    data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
    data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
    data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
    data1["degree_t"]=le.fit_transform(data1["degree_t"])
    data1["workex"]=le.fit_transform(data1["workex"])
    data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
    data1["status"]=le.fit_transform(data1["status"])       
    data1 
    
    x=data1.iloc[:,:-1]
    x
    y=data1["status"]
    y
    
    from sklearn.model_selection import train_test_split
    x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)
    
    from sklearn.linear_model import LogisticRegression
    lr=LogisticRegression(solver="liblinear")
    lr.fit(x_train,y_train)
    y_pred=lr.predict(x_test)
    y_pred
    
    from sklearn.metrics import accuracy_score
    accuracy=accuracy_score(y_test,y_pred)
    accuracy
    
    from sklearn.metrics import confusion_matrix
    confusion=confusion_matrix(y_test,y_pred)
    confusion
    
    from sklearn.metrics import classification_report
    classification_report1 = classification_report(y_test,y_pred)
    print(classification_report1)
    lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])
```

## Output:

<img width="1259" height="229" alt="Screenshot 2025-09-27 110725" src="https://github.com/user-attachments/assets/84811870-90ce-4407-a1f9-bfa73aecdaa7" />

<img width="1263" height="274" alt="Screenshot 2025-09-27 110737" src="https://github.com/user-attachments/assets/69eb6c15-1782-457e-b891-b80af21da725" />

<img width="1224" height="617" alt="Screenshot 2025-09-27 110753" src="https://github.com/user-attachments/assets/9635d595-0b05-4712-88b5-c951f47ceee7" />

# Data duplicates

<img width="70" height="57" alt="Screenshot 2025-09-27 110800" src="https://github.com/user-attachments/assets/1b55db75-dbf9-4a53-a8cb-5e0bdfd3fa88" />

# Print data 

<img width="1219" height="626" alt="Screenshot 2025-09-27 110813" src="https://github.com/user-attachments/assets/429888a8-d2d3-4b18-ac3e-c1a64bdd48b6" />

# Data-status

<img width="1148" height="634" alt="Screenshot 2025-09-27 110822" src="https://github.com/user-attachments/assets/521b3a52-046a-45e2-b7cc-9d59ef31bdef" />

# y_prediction array:

<img width="729" height="326" alt="Screenshot 2025-09-27 110832" src="https://github.com/user-attachments/assets/3ddc905f-a5b1-4d7a-b9a2-89ea68f89518" />

# Confusion array:

<img width="945" height="81" alt="Screenshot 2025-09-27 110841" src="https://github.com/user-attachments/assets/face2c54-679c-433b-9d68-268b13c3b7a3" />

# Accuracy Value:

<img width="255" height="62" alt="Screenshot 2025-09-27 110852" src="https://github.com/user-attachments/assets/aff85f89-368f-4af0-ae3a-0bfca94eace1" />

# Classification Report:

<img width="710" height="212" alt="Screenshot 2025-09-27 110859" src="https://github.com/user-attachments/assets/5f381566-1063-40c8-8aad-4e00b3527f6e" />

# Prediction of LR:

<img width="327" height="34" alt="Screenshot 2025-09-27 110914" src="https://github.com/user-attachments/assets/08fef835-7be6-4fd4-9533-7b1c01d76f9c" />



## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
