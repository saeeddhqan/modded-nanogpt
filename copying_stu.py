import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import random
import string
import stu
from model import Model

def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(1234)

# Define dataset for copying task
class CopyingTaskDataset(Dataset):
    def __init__(self, num_samples=10000, seqlen=10, vocab_size=27, reverse: bool = False):
        self.num_samples = num_samples
        self.seqlen = seqlen // 2
        self.vocab_size = vocab_size
        self.char_to_idx = {char: idx for idx, char in enumerate(string.ascii_lowercase)}
        self.idx_to_char = {idx: char for char, idx in self.char_to_idx.items()}
        self.data = self._generate_data()
        self.reverse = reverse

    def _generate_data(self):
        data = []
        for _ in range(self.num_samples):
            sequence = [random.randint(0, self.vocab_size - 1) for _ in range(self.seqlen)]
            data.append(sequence)
        return data

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        """
            input:  1234____
            target: ____1234
            reverse case:
            input:  1234____
            target: ____4321
        """
        data = self.data[idx]
        input_seq = data + [self.vocab_size  for _ in range(self.seqlen)]
        if self.reverse:
            data = data[::-1]
        target_seq = [self.vocab_size  for _ in range(self.seqlen)] + data
        input_seq = torch.tensor(input_seq, dtype=torch.long)
        target_seq = torch.tensor(target_seq, dtype=torch.long)
        return input_seq, target_seq


# Initialize model, loss, optimizer
batch_size = 64
num_epochs = 10

vocab_size = 10 # Lowercase letters
embed_dim = 32
seqlen = 32
nlayers = 10

K = 8
num_slots = 4
use_gating = False

reverse_case = True
device = 'cuda'
method = 'attn'
dtype = torch.float32
is_causal = False
# model = stu.STUModel(dim=embed_dim, seqlen=seqlen, nlayers=nlayers, K=K,
    # num_slots=num_slots, vocab_size=vocab_size + 1, use_gating=use_gating, device=device)
model = Model(vocab_size=vocab_size + 1, num_layers=nlayers, model_dim=embed_dim, method=method, num_heads=2, seqlen=seqlen, is_causal=is_causal, K=K,
    num_slots=num_slots, use_gating=use_gating, device=device, dtype=dtype)
model.to(device)
optimizer = optim.AdamW(model.parameters(), lr=1e-3)



# Prepare dataset and dataloaders
train_dataset = CopyingTaskDataset(num_samples=20000, seqlen=seqlen, vocab_size=vocab_size, reverse=reverse_case)
# train_dataset = CopyingTaskDataset(num_samples=10, seqlen=seqlen, vocab_size=vocab_size)
test_dataset = CopyingTaskDataset(num_samples=2000, seqlen=seqlen, vocab_size=vocab_size, reverse=reverse_case)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Training loop
def train_model():
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for input_seq, target_seq in train_loader:
            input_seq, target_seq = input_seq.to(device), target_seq.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, _= model(input_seq, target_seq)  # Shape: (batch_size, seqlen, vocab_size)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0,
            )
            optimizer.step()

            total_loss += loss.item()
            print(loss.item(), end='\r')
        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_loader):.4f}")

# Evaluation loop
def evaluate_model():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for input_seq, target_seq in test_loader:
            input_seq, target_seq = input_seq.to(device), target_seq.to(device)
            _, logits = model(input_seq)
            predictions = torch.argmax(logits, dim=-1)  # Get most probable tokens

            correct += (predictions == target_seq).sum().item()
            total += target_seq.numel()
    
    accuracy = correct / total
    print(f"Copying Task Accuracy: {accuracy * 100:.2f}%")

# Run training and evaluation
train_model()
evaluate_model()
