import numpy as np


x = np.array([[2,3], [4,1], [5,2]])
y = np.array([10, 13, 16])

X_b = np.c_[np.ones((x.shape[0], 1)), x]

theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)

print("پارامترهای بهینه (theta):", theta_best)