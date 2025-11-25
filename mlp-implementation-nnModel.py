import torch
import torch.nn as nn
import torch.optim as optim

# --------------------------
# 1. تعریف مدل با nn.Module
# --------------------------

class MyMLP(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(4, 8)   # لایه اول
        self.fc2 = nn.Linear(8, 6)   # لایه دوم
        self.fc3 = nn.Linear(6, 4)   # لایه سوم
        self.fc4 = nn.Linear(4, 2)   # لایه خروجی

    def forward(self, x):
        x = torch.relu(self.fc1(x))   # Activation 1
        x = torch.relu(self.fc2(x))   # Activation 2
        x = torch.sigmoid(self.fc3(x))  # Activation 3
        x = self.fc4(x)               # لایه خروجی (بدون softmax)
        return x

# ساخت مدل
model = MyMLP()


# --------------------------
# 2. دیتای ورودی و خروجی
# --------------------------

x = torch.randn(1, 4)     # یک ورودی با 4 ویژگی
target = torch.tensor([1])  # کلاس هدف (کلاس شماره 1)


# --------------------------
# 3. Loss و Optimizer
# --------------------------

criterion = nn.MSELoss() #Loss → خطای پیش‌بینی
optimizer = optim.SGD(model.parameters(), lr=0.1) #Optimizer → آپدیت وزن‌ها با گرادیان


for epoch in range(10):
    optimizer.zero_grad()          # صفر کردن گرادیان‌ها
    y_pred = model(x)              # Forward Pass
    loss = criterion(y_pred, y)    # محاسبه خطا
    loss.backward()                # Backpropagation خودکار
    optimizer.step()               # آپدیت وزن‌ها
    print(f"Epoch {epoch+1}: Loss = {loss.item()}")
