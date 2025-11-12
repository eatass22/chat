import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import re

# ============================================================
# CONFIG
# ============================================================
DATA_DIR = "datasets"
MODEL_PATH = "lewitch_word_transformer.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 15
BATCH_SIZE = 16
SEQ_LEN = 10
EMBED_DIM = 64
NUM_HEADS = 4
NUM_LAYERS = 2
LR = 1e-3

# ============================================================
# LOAD ALL TEXT FILES
# ============================================================
def load_all_texts(data_dir):
    text_data = ""
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        open(os.path.join(data_dir, "example.txt"), "w").write(
            "hello world this is a sample file for the Lewitch AI."
        )
    for file in os.listdir(data_dir):
        if file.endswith(".txt"):
            with open(os.path.join(data_dir, file), "r", encoding="utf-8") as f:
                text_data += f.read() + "\n"
    return text_data.strip()

# ============================================================
# DATASET
# ============================================================
class WordDataset(Dataset):
    def __init__(self, text, seq_len):
        words = re.findall(r"\b\w+\b", text.lower())
        self.vocab = sorted(set(words))
        self.word2idx = {w: i for i, w in enumerate(self.vocab)}
        self.idx2word = {i: w for w, i in self.word2idx.items()}
        self.seq_len = seq_len
        self.data = [self.word2idx[w] for w in words]

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        seq = torch.tensor(self.data[idx:idx + self.seq_len])
        target = torch.tensor(self.data[idx + 1:idx + self.seq_len + 1])
        return seq, target

# ============================================================
# MODEL
# ============================================================
class WordTransformer(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Embedding(512, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            embed_dim, num_heads, dim_feedforward=512, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, x):
        positions = torch.arange(0, x.size(1), device=x.device).unsqueeze(0)
        x = self.embed(x) + self.pos_embed(positions)
        x = self.transformer(x)
        return self.fc_out(x)

# ============================================================
# TRAIN LOOP
# ============================================================
def train_model(model, dataloader, epochs, lr):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for x, y in dataloader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output.view(-1, output.size(-1)), y.view(-1))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/len(dataloader):.4f}")
    return model

# ============================================================
# GENERATE TEXT
# ============================================================
def generate(model, dataset, prompt, max_len=20):
    model.eval()
    words = re.findall(r"\b\w+\b", prompt.lower())
    input_seq = [dataset.word2idx.get(w, 0) for w in words][-SEQ_LEN:]
    generated = []

    for _ in range(max_len):
        x = torch.tensor(input_seq, dtype=torch.long).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            out = model(x)
            probs = torch.softmax(out[0, -1], dim=-1)
            next_token = torch.multinomial(probs, num_samples=1).item()
        next_word = dataset.idx2word[next_token]
        generated.append(next_word)
        input_seq = (input_seq + [next_token])[-SEQ_LEN:]

    return " ".join(generated)

# ============================================================
# MAIN
# ============================================================
if __name__ == "__main__":
    text = load_all_texts(DATA_DIR)
    dataset = WordDataset(text, SEQ_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    vocab_size = len(dataset.vocab)
    print(f"✅ Vocab size: {vocab_size}")

    model = WordTransformer(vocab_size, EMBED_DIM, NUM_HEADS, NUM_LAYERS).to(DEVICE)

    if os.path.exists(MODEL_PATH):
        print("Loading existing model...")
        try:
            model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
        except RuntimeError:
            print("⚠️ Vocab mismatch — retraining model...")
            model = train_model(model, dataloader, EPOCHS, LR)
            torch.save(model.state_dict(), MODEL_PATH)
    else:
        print("Model not found — training new one...")
        model = train_model(model, dataloader, EPOCHS, LR)
        torch.save(model.state_dict(), MODEL_PATH)
        print("💾 Model saved!")

    # ============================================================
    # CHAT LOOP
    # ============================================================
    print("\n💬 Chat with your Lewitch AI (type 'quit' to stop)")
    while True:
        prompt = input("> ").strip()
        if prompt.lower() == "quit":
            break
        reply = generate(model, dataset, prompt, max_len=15)
        print(reply)
