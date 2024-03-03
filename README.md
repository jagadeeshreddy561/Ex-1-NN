 ## NAME : jagadeesh Reddy
## REGISTER NO : 212222240059
## EX. NO.1
## DATE : 03/03/2024
## Introduction to Kaggle and Data preprocessing

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
## STEP 1:
Importing the libraries<BR>
## STEP 2:
Importing the dataset<BR>
## STEP 3:
Taking care of missing data<BR>
## STEP 4:
Encoding categorical data<BR>
## STEP 5:
Normalizing the data<BR>
## STEP 6:
Splitting the data into test and train<BR>

##  PROGRAM:
## Import Libraries
from google.colab import files
import pandas as pd
import seaborn as sns
import io
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy import stats
import numpy as np
## Read the dataset
df=pd.read_csv("Churn_Modelling.csv")
## Checking Data
df.head()
df.tail()
df.columns
## Check the missing data
df.isnull().sum()
## Check for Duplicates
df.duplicated()
## Assigning Y
y = df.iloc[:, -1].values
print(y)
## Check for duplicates
df.duplicated()
## Check for outliers
df.describe()
## Dropping string values data from dataset
data = df.drop(['Surname', 'Geography','Gender'], axis=1)
## Checking datasets after dropping string values data from dataset
data.head()
## Normalize the dataset
scaler=MinMaxScaler()
df1=pd.DataFrame(scaler.fit_transform(data))
print(df1)
## Split the dataset
X=df.iloc[:,:-1].values
y=df.iloc[:,-1].values
print(X)
print(y)
## Training and testing model
X_train ,X_test ,y_train,y_test=train_test_split(X,y,test_size=0.2)
print("X_train\n")
print(X_train)
print("\nLenght of X_train ",len(X_train))
print("\nX_test\n")
print(X_test)
print("\nLenght of X_test ",len(X_test))

## OUTPUT:
## Data checking

![exp - 1 1](https://github.com/jagadeeshreddy561/Ex-1-NN/assets/120623104/69b82b48-47d3-412c-b739-e10dd27fbc29)


## Missing Data


![exp-1 2](https://github.com/jagadeeshreddy561/Ex-1-NN/assets/120623104/cb042ea2-adb5-4476-abbc-93ba62d4cd56)


## Duplicates identification

![exp-1 3](https://github.com/jagadeeshreddy561/Ex-1-NN/assets/120623104/ab764e4f-57a8-439c-a5c9-cea69c1b7bd3)


## Vakues of 'Y'


![exp-1 4](https://github.com/jagadeeshreddy561/Ex-1-NN/assets/120623104/ebb950a9-2a02-44d5-bace-2c64e2b7147f)


## Outliers


![exp-1 5](https://github.com/jagadeeshreddy561/Ex-1-NN/assets/120623104/c3421fdd-3062-40b1-96e4-b052e181b0e4)


## Checking datasets after dropping string values data from dataset


![exp-1 6](https://github.com/jagadeeshreddy561/Ex-1-NN/assets/120623104/7cdb00a0-efdc-454d-b0dd-92e131a180c2)


## Normalize the dataset


![exp-1 7](https://github.com/jagadeeshreddy561/Ex-1-NN/assets/120623104/9aa31804-38ca-44ba-a38f-77f2936e197b)


## Split the dataset


![exp-1 8](https://github.com/jagadeeshreddy561/Ex-1-NN/assets/120623104/886c69d3-8ca2-42ed-a675-583a2b49e0ff)

## Training and testing model


![exp-1 9](https://github.com/jagadeeshreddy561/Ex-1-NN/assets/120623104/b41f5931-dd79-415c-a181-b8438e188bb6)



## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


