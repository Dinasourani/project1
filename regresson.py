import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import linear_model
import sklearn.metrics as sm
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


df= pd.read_csv('co2.csv')
print(df.describe()) #نشون میده دیتا هارو



sns.countplot(x='out1',data=df) #امار و تعداد خروجی خوددمون رو میتونستیم ببینیم

plt.subplots(figsize=(4,4))
sns.heatmap(df.corr(), annot=True) #یک heatmap که میگه برای این ورودی خروجی مرتبط اون چقدر تاثیر داره
#مثلا out1--> engine=0.88


x = df.drop("out1",axis=1)
y = df.out1

X_train,X_test,y_train,y_test=train_test_split(x,y, test_size=0.2)

model=linear_model.LinearRegression()
model.fit(X_train,y_train)
out_robot=model.predict(X_test)


plt.scatter(X_train.engine,y_train,color="red")
plt.scatter(X_train.fuelcomb,y_train,color="green")
plt.scatter(X_train.cylandr,y_train,color="blue")
plt.show()


p209=np.array([[2,4,9.8]])
co2=model.predict(p209)


plt.show()