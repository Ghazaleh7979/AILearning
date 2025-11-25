import numpy as np

x = np.array([[0.1, 0.2, 0.3]])

# وزن های لایه پنهان
w1 = np.random.randn(3, 4) * 0.1 # (input_dim, hidden_dim)
b1 = np.zeros((1, 4))

# وزن های لایه خروجی
w2 = np.random.randn(4, 2) * 0.1 # (hidden_dim, output_dim)
b2 = np.zeros((1, 2))

def relu(z):
    return np.maximum(0, z)

def softmax(z):
    exp = np.exp(z - np.max(0, z))
    return exp / exp.sum(axis=1, keepdims=True)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# لایه پنهان
z1 = x @ w1 + b1
a1 = relu(z1)

# لایه خروجی
z2 = a1 @ w2 + b2
output = softmax(z2)

print("Hidden layer activation:", a1)
print("Network output:", output)
