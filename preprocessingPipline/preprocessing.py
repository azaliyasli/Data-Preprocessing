import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("C:\\Users\\slkn5\\Desktop\\ML\\Resources\\Datasets\\Data1.csv")
X = df.iloc[:, :-1].values  #All rows and all columns except last column
y = df.iloc[:, -1].values  #All rows and just last column

print(X)
print(y)

print("-----Filling Missing Values-----")
imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])
print(X)

print("-----Turning Categorical Data Into Numerical Data (Encoding Independent Variables)-----")
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
print(X)

print("-----Encoding Dependent Variables-----")
le = LabelEncoder()
y = le.fit_transform(y)
print(y)

print("-----Training and Testing-----")
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42) #These values are general
sc = StandardScaler()
X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
X_val[:, 3:] = sc.transform(X_val[:, 3:])
print(f"""Training: 
{X_train}
Validation: 
{X_val}""")








