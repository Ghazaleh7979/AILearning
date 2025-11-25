import torch
import torch.nn as nn
import torch.optim as optim
 
x = torch.randn(5, 4)  # 4 samples, 4 features each
y = torch.tensor([0, 1, 0, 1, 1])  # 4 samples, binary target

# --------------------------
# 2. ساخت وزن‌ها (requires_grad=True)
# --------------------------
#تعداد ورودی‌های هر لایه = تعداد نورون‌های لایه قبلی
W1 = torch.randn(4, 8, requires_grad=True)   # لایه اول
b1 = torch.randn(8, requires_grad=True)

W2 = torch.randn(8, 6, requires_grad=True)   # لایه دوم
b2 = torch.randn(6, requires_grad=True)

W3 = torch.randn(6, 4, requires_grad=True)   # لایه سوم
b3 = torch.randn(4, requires_grad=True)

W4 = torch.randn(4, 2, requires_grad=True)   # لایه خروجی
b4 = torch.randn(2, requires_grad=True)

# --------------------------
# 3. تابع فعال‌سازی
# --------------------------

def sigmoid(x):
    return 1 / (1 + torch.exp(-x))


# --------------------------
# 4. Forward Pass
# --------------------------

# Hidden Layer 1
z1 = x @ W1 + b1
a1 = sigmoid(z1)

# Hidden Layer 2
z2 = a1 @ W2 + b2
a2 = sigmoid(z2)

# Hidden Layer 3
z3 = a2 @ W3 + b3
a3 = sigmoid(z3)

# Output Layer (logits)
z4 = a3 @ W4 + b4


# --------------------------
# 5. Loss (CrossEntropy)
# --------------------------
#اینجا رو خروجی softmax میزنه خودش
loss = torch.nn.functional.cross_entropy(z4, y)
print("Loss:", loss.item())

# --------------------------
# 6. Backpropagation
# --------------------------
#Backpropagation + Chain rule تمام مشتق‌ها را از خروجی تا وزن‌ها محاسبه می‌کند
loss.backward()

# حالا گرادیان‌ها در .grad هستند
print(W1.grad)
print(W2.grad)
print(W3.grad)
print(W4.grad)

# --------------------------
# 7. Update Rule (GD ساده)
# --------------------------

lr = 0.01

with torch.no_grad():
    W1 -= lr * W1.grad
    b1 -= lr * b1.grad
    W2 -= lr * W2.grad
    b2 -= lr * b2.grad
    W3 -= lr * W3.grad
    b3 -= lr * b3.grad
    W4 -= lr * W4.grad
    b4 -= lr * b4.grad

    # صفر کردن گرادیان‌ها
    W1.grad.zero_()
    b1.grad.zero_()
    W2.grad.zero_()
    b2.grad.zero_()
    W3.grad.zero_()
    b3.grad.zero_()
    W4.grad.zero_()
    b4.grad.zero_()