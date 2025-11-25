import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = load_iris()
X = iris.data[:, :2]
y = iris.target


mask = y < 2
X = X[mask]
y = y[mask]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#مقیاس بندی داده هایترین و اعمال روی داده های تست
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#توی فیت ازچیز هایی مثل گرادیان دسنت استفاده میکنه و وزن هارو پیدا میکنه
model = LogisticRegression()
model.fit(X_train, y_train)

#Sigmoid تو اینه
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)


w = model.coef_[0]
b = model.intercept_[0]

x_values = np.linspace(-2, 2, 100)
y_values = -(w[0] * x_values + b) / w[1]

plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis', edgecolors='k')
plt.plot(x_values, y_values, color='red')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Logistic Regression Decision Boundary')
plt.show()