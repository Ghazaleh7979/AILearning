import torch
import torch.nn as nn
import torch.optim as optim

# --------- 1) داده ---------
# A B C → D
seq = torch.tensor([[0, 1, 2]])     # ورودی (ABC)
target = torch.tensor([3])          # خروجی (D)

vocab_size = 4 #تعداد کل حروف (A, B, C, D).
embed_dim = 4 # بعد بردار جاسازی (Embedding) هر حرف، یعنی هر حرف با یک بردار 4 بعدی نمایش داده می‌شود.
hidden_size = 8 #تعداد نورون‌ها در لایه RNN، یعنی شبکه حافظه کوچکی برای دنباله دارد.


# --------- 2) مدل RNN ---------
class SmallRNN(nn.Module): #همه شبکه‌ها در PyTorch از nn.Module ارث‌بری می‌کنند.
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim) #Embedding حرف را به یک بردار عددی تبدیل می‌کند (مثلاً A → [0.1, -0.2, 0.5, 0.7]).
        self.rnn = nn.RNN(input_size=embed_dim, hidden_size=hidden_size, batch_first=True)  #لایه RNN (شبکه عصبی بازگشتی).
        #batch_first=True → شکل ورودی [batch, seq_len, feature] باشد، نه [seq_len, batch, feature].
        self.fc = nn.Linear(hidden_size, vocab_size) #آخرین مرحله: تبدیل خروجی RNN به احتمال هر حرف.

    def forward(self, x):
        x = self.embedding(x)                # تبدیل اعداد به بردارهای عددی.
        out, h = self.rnn(x)                 # (batch, seq_len, hidden_size)
        out = self.fc(out[:, -1, :])         # فقط آخرین خروجی → پیش‌بینی حرف بعدی
        return out


model = SmallRNN()
loss_fn = nn.CrossEntropyLoss() #تابع خطا برای طبقه‌بندی چندکلاسه.
optimizer = optim.Adam(model.parameters(), lr=0.01) #الگوریتم آدمین برای بروزرسانی وزن‌ها.
#lr=0.01 سرعت یادگیری است.


# --------- 3) آموزش خیلی کوچک ---------
for epoch in range(200):
    optimizer.zero_grad() # صفر کردن گرادیان‌ها
    output = model(seq) # پیش‌بینی شبکه
    loss = loss_fn(output, target) # محاسبه خطا
    loss.backward() # محاسبه گرادیان‌ها
    optimizer.step()  # آپدیت وزن‌ها

    if epoch % 50 == 0:
        print(epoch, loss.item()) #هر 50 دوره (epoch) یکبار خطا را چاپ می‌کنیم تا ببینیم شبکه یاد می‌گیرد یا نه.


# --------- 4) تست ---------
with torch.no_grad(): #می‌گوید اینجا نیازی به محاسبه گرادیان‌ها نیست (برای تست و پیش‌بینی).
    out = model(seq)
    pred = torch.argmax(out) #حرف با بیشترین احتمال را انتخاب می‌کنیم.
    print("Predicted:", pred.item())  #تبدیل tensor به عدد ساده.
