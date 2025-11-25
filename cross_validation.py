from sklearn.model_selection import KFold, cross_val_score
from sklearn.linear_model import LinearRegression
import numpy as np

x = np.random.randn(100, 1)
y = 3 * x[:, 0] + 2 + np.random.randn(100) * 0.1

model = LinearRegression()
kf = KFold(n_splits=5, shuffle=True, random_state=42)

mse_scores = cross_val_score(model, x, y, cv=kf, scoring='neg_mean_squared_error')
mae_scores = cross_val_score(model, x, y, cv=kf, scoring='neg_mean_absolute_error')
r2_scores = cross_val_score(model, x, y, cv=kf, scoring='r2')

mse_scores = -mse_scores
mae_scores = -mae_scores

print("MSE for each fold:", mse_scores)
print("Mean MSE:", mse_scores.mean())

print("MAE for each fold:", mae_scores)
print("Mean MAE:", mae_scores.mean())

print("R² for each fold:", r2_scores)
print("Mean R²:", r2_scores.mean())