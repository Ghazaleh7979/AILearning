import random, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence
random.seed(1); torch.manual_seed(1)

subjects = ["من","تو","او"]
verbs = ["رفتم","می‌روم","دیدم"]
objects = ["مدرسه","کتاب","بازار"]
sentences = []

for _ in range(200):  # کمتر نمونه برای سرعت
    s = random.choice(subjects); v = random.choice(verbs); o = random.choice(objects)
    sentences.append(f"{s} {v} {o}")
    
sentences += ["من به مدرسه رفتم", "او کتاب دید"]

# 2) vocab و نگاشت‌ها
PAD, UNK, BOS, EOS = "<pad>", "<unk>", "<bos>", "<eos>"
all_tokens = [PAD, UNK, BOS, EOS]
word_set = set()
for s in sentences:
    for w in s.split(): word_set.add(w)
vocab_list = all_tokens + sorted(word_set)
w2i = {w:i for i,w in enumerate(vocab_list)}
i2w = {i:w for w,i in w2i.items()}

# 3) جفت‌های (prefix -> next_word)
pairs = []
for s in sentences:
    toks = [BOS] + s.split() + [EOS]
    ids = [w2i.get(t,w2i[UNK]) for t in toks]
    for t in range(len(ids)-1):
        inp = torch.tensor(ids[:t+1], dtype=torch.long)
        tgt = ids[t+1]
        pairs.append((inp, tgt))
        
# 4) Dataset / DataLoader (with padding)
class NW(Dataset):
    def __init__(self,pairs): self.pairs = pairs
    def __len__(self): return len(self.pairs)
    def __getitem__(self,idx): return self.pairs[idx]
    

def collate(batch):
    inputs = [item[0] for item in batch]
    targets = torch.tensor([item[1] for item in batch], dtype=torch.long)
    lengths = torch.tensor([len(x) for x in inputs], dtype=torch.long)
    padded = pad_sequence(inputs, batch_first=True, padding_value=w2i[PAD])
    return padded, lengths, targets

loader = DataLoader(NW(pairs), batch_size=32, shuffle=True, collate_fn=collate)

# 5) مدل کوچک LSTM
class SmallWordLSTM(nn.Module):
    def __init__(self, vocab, emb=32, hid=64):
        super().__init__()
        self.embedding = nn.Embedding(vocab, emb, padding_idx=w2i[PAD])
        self.lstm = nn.LSTM(emb, hid, batch_first=True)
        self.fc = nn.Linear(hid, vocab)
    def forward(self, x, lengths):
        emb = self.embedding(x)
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h,_) = self.lstm(packed)
        last = h[-1]
        return self.fc(last)
    
    
    
model = SmallWordLSTM(len(vocab_list))
opt = torch.optim.Adam(model.parameters(), lr=0.01)
crit = nn.CrossEntropyLoss()

# 6) آموزش سریع (چند epoch)
for epoch in range(1, 11):  # فقط 10 epoch برای سرعت
    total_loss = 0
    model.train()
    for xb, lengths, yb in loader:
        opt.zero_grad()
        logits = model(xb, lengths)
        loss = crit(logits, yb)
        loss.backward()
        opt.step()
        total_loss += loss.item()*xb.size(0)

def predict(prompt, topk=5):
    words = prompt.split() if prompt.strip() else []
    toks = [BOS] + words
    ids = [w2i.get(t, w2i[UNK]) for t in toks]
    x = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    lengths = torch.tensor([len(ids)])
    with torch.no_grad():
        logits = model(x, lengths)
        probs = torch.softmax(logits, dim=-1).squeeze(0)
    topv, topi = torch.topk(probs, k=min(topk, probs.size(0)))
    return [(i2w[int(i)], float(v)) for i,v in zip(topi.tolist(), topv.tolist())]
    
print("=== Mini interactive next-word predictor ===")
print("برای خروج، فقط 'exit' را تایپ کن.\n")

prompt = ""
while True:
    prompt = input("جمله فعلی: ")
    if prompt.strip().lower() == "exit":
        break
    predictions = predict(prompt, topk=5)
    print("کلمات پیشنهادی با احتمال:")
    for word, prob in predictions:
        print(f"  {word} ({prob:.2f})")
    print()