import numpy as np

x = np.array(1,2,3)
y = np.array(2,3,6)

n = len(x)


x_b = np.c_[np.ones((n, 1)), x]

theta = np.zeros(2)

alpha = 0.01
iterations = 1000

for _ in range(iterations):
    gradients = 2/n * x_b.T.dot(x_b.dot(theta) - y)
    theta -= alpha * gradients
    
print("پارامترهای بهینه (theta):", theta)