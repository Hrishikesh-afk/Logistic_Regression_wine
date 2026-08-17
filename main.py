%pip install seaborn
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
df=pd.read_csv("Wine dataset.csv")
df.info()

df.describe()

df.head()

X=df.drop("class",axis=1)
y=df["class"]
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)

from sklearn.linear_model import LogisticRegression
model=LogisticRegression(max_iter=5000)
model.fit(X_train,y_train)
y_pred=model.predict(X_test)

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
accuracy=accuracy_score(y_test,y_pred)
print("Accuracy Score:",accuracy)
confusion=confusion_matrix(y_test,y_pred)
print("Confusion Matrix:\n",confusion)
classification=classification_report(y_test,y_pred)
print("Classification Report:\n",classification)
