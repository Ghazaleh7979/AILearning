import numpy as np
# -------- داده ----------
x = np.random.rand(100, 1)
y = 3 * x[:, 0] + 2 + np.random.randn(100) * 0.1

#۸۰٪ برای آموزش، ۲۰٪ برای تست
from sklearn.model_selection import train_test_split
x_train, x_test, y_train , y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# -------- Lasso Regression ----------
from sklearn.linear_model import Lasso
lasso_model = Lasso(alpha=0.1)
lasso_model.fit(x_train, y_train)
print("ضریب (slope):", lasso_model.coef_[0])
print("عرض از مبدأ (intercept):", lasso_model.intercept_)

#MSE
y_pred = lasso_model.predict(x_test)
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
print("MSE روی داده تست:", mse)



