import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# seed برای تکرارپذیری
torch.manual_seed(42)

# dataset خیلی کوچک: توالی‌های 0 تا 9
seq_len = 10
data = [list(range(i, i+seq_len)) for i in range(30)]  # مثال: [[0,1,2,3,4], [1,2,3,4,5], ...]

# Dataset و DataLoader
class SeqDataset(Dataset):
    def __init__(self, data):
        self.data = data
    def __len__(self): return len(self.data)
    def __getitem__(self, idx):
        x = torch.tensor(self.data[idx][:-1], dtype=torch.float).unsqueeze(-1) # همه جز آخر
        y = torch.tensor(self.data[idx][-1], dtype=torch.float)                # آخرین عنصر
        return x, y

loader = DataLoader(SeqDataset(data), batch_size=1, shuffle=True)

class SimpleRNN(nn.Module):
    def __init__(self, rnn_type='RNN', input_size=1, hidden_size=10):
        super().__init__()
        if rnn_type == 'RNN':
            self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        elif rnn_type == 'GRU':
            self.rnn = nn.GRU(input_size, hidden_size, batch_first=True)
        elif rnn_type == 'LSTM':
            self.rnn = nn.LSTM(input_size, hidden_size, batch_first=True)
        else:
            raise ValueError("rnn_type must be RNN, GRU or LSTM")
        self.fc = nn.Linear(hidden_size, 1)
        self.rnn_type = rnn_type

    def forward(self, x):
        if self.rnn_type == 'LSTM':
            out, (h,c) = self.rnn(x)
        else:
            out, h = self.rnn(x)
        # استفاده از آخرین hidden state برای پیش‌بینی
        return self.fc(h[-1])
    
def train_model(model, loader, epochs=80):
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    for epoch in range(epochs):
        total_loss = 0
        for xb, yb in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred.squeeze(), yb.squeeze())
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

# RNN
rnn_model = SimpleRNN('RNN')
train_model(rnn_model, loader)

# GRU
gru_model = SimpleRNN('GRU')
train_model(gru_model, loader)

# LSTM
lstm_model = SimpleRNN('LSTM')
train_model(lstm_model, loader)



x_test = torch.tensor([[1,2,3,4]], dtype=torch.float).unsqueeze(-1)
print("RNN pred:", rnn_model(x_test).item())
print("GRU pred:", gru_model(x_test).item())
print("LSTM pred:", lstm_model(x_test).item())
