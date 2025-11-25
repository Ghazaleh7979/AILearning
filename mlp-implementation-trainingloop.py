import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# --------------------------
# 1. داده‌ها و DataLoader
# --------------------------
x = torch.randn(100, 3)  # 100 نمونه، 3 ویژگی
y = torch.randint(0, 2, (100,))  # برچسب دودویی

dataset = TensorDataset(x, y)
dataloader = DataLoader(dataset, batch_size=16, shuffle=True)

# --------------------------
# 2. تعریف مدل
# --------------------------
class MyMLP(nn.Module):
    def __init__(self):
        super(MyMLP, self).__init__()
        self.hidden1 = nn.Linear(3, 8)
        self.hidden2 = nn.Linear(8, 4)
        self.output = nn.Linear(4, 2)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        h1 = self.relu(self.hidden1(x))
        h2 = self.relu(self.hidden2(h1))
        out = self.output(h2)
        return out

model = MyMLP()

# --------------------------
# 3. Loss و Optimizer
# --------------------------
criterion = nn.CrossEntropyLoss()         # مناسب طبقه‌بندی چندکلاسه
optimizer = optim.Adam(model.parameters(), lr=0.01)

# --------------------------
# 4. Training Loop
# --------------------------
epochs = 10
for epoch in range(epochs):
    running_loss = 0.0
    for batch_x, batch_y in dataloader:
        optimizer.zero_grad()          # صفر کردن گرادیان‌ها
        outputs = model(batch_x)       # Forward Pass
        loss = criterion(outputs, batch_y)  # محاسبه خطا
        loss.backward()                # Backprop
        optimizer.step()               # آپدیت وزن‌ها
        
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(dataloader):.4f}")
