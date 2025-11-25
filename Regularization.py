import numpy as np
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


np.random.seed(42)

X = np.random.randn(100, 10)

true_coef = np.array([1.5, -2.0, 0.0, 0.0, 0.5, 0, 0, 0, 0, 0])

y = X @ true_coef + np.random.randn(100) * 1.0 


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
print("Linear Regression coefficients:", lin_reg.coef_)
print("Linear Regression intercept:", lin_reg.intercept_)
y_pred = lin_reg.predict(X_test)
mse_test = mean_squared_error(y_test, y_pred)
print("MSE روی داده تست:", mse_test)

ridge_reg = Ridge(alpha=0.1)
ridge_reg.fit(X_train, y_train)
print("Ridge Regression coefficients:", ridge_reg.coef_)
y_pred = ridge_reg.predict(X_test)
mse_test = mean_squared_error(y_test, y_pred)
print("MSE روی داده تست:", mse_test)


lasso_reg = Lasso(alpha=0.1)
lasso_reg.fit(X_train, y_train)
print("Lasso Regression coefficients:", lasso_reg.coef_)
y_pred = lasso_reg.predict(X_test)
mse_test = mean_squared_error(y_test, y_pred)
print("MSE روی داده تست:", mse_test)

