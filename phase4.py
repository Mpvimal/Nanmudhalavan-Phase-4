import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn.tree import DecisionTreeRegressor

data = pd.read_csv('Sales.csv')
print("Description of columns : \n")
print(data.describe())  #decription of each column
print("\nNo.of Null Columns :\n",data.isnull().sum())  #count of null values in columns
data = data.dropna()    #to remove null data
print(data.corr())
correlations = data.corr(method='pearson')
plt.figure(figsize=(15, 12))
sns.heatmap(correlations, cmap="coolwarm", annot=True)
plt.show()
x=data[["TV","Radio","Newspaper"]]
y=data["Sales"]
xtrain, xtest, ytrain, ytest = train_test_split(x, y,test_size=0.2,random_state=42)
model = DecisionTreeRegressor()
model.fit(xtrain, ytrain)
features = np.array([[200,40,100]])
us=model.predict(features)
print(us)
