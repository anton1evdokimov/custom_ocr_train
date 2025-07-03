import re
import torch
import torch.nn as nn

from OCRDataset import OCRDataset

import torch.optim as optim
from torch.utils.data import DataLoader

from const import IMG_HEIGHT
from model import CRNN
from utils import alphabet, decode, get_samples, num_classes, transform, device

samples = get_samples(root = 'сrop_images')

model_name = 'model_loss_0.00007.pth'
model = CRNN(imgH=IMG_HEIGHT, nc=1, nclass=num_classes, nh=256).to(device)
model.load_state_dict(torch.load(model_name))

optimizer = optim.Adam(model.parameters(), lr=1e-3)

ctc_loss = nn.CTCLoss(blank=len(alphabet))

min_loss = float(re.findall(r'\d+(?:\.\d+)', model_name)[0])

dataset = OCRDataset(samples=samples, transform=transform)
loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda x: x)

for epoch in range(20):
    model.train()
    for batch in loader:
        imgs, labels, label_lengths = zip(*batch)

        # Prepare tensors
        imgs = torch.stack(imgs).to(device)
        labels_concat = torch.cat(labels).to(device)
        label_lengths = torch.tensor(label_lengths, dtype=torch.long).to(device)

        preds = model(imgs)  # (T, B, nclass)
        res = decode(preds)
        # CTC requires input lengths
        T = preds.size(0)
        pred_lengths = torch.full(size=(preds.size(1),), fill_value=T, dtype=torch.long).to(device)

        loss = ctc_loss(preds.log_softmax(2), labels_concat, pred_lengths, label_lengths)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch}, Loss: {loss.item():.5f}")
    if loss.item() < min_loss:
        min_loss = loss.item()
        torch.save(model.state_dict(), f"model_loss_{loss.item():.5f}.pth")