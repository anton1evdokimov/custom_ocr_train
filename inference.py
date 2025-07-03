

import torch
from PIL import Image
from const import IMG_HEIGHT
from model import CRNN

from utils import decode, get_samples, num_classes, device, transform

samples = get_samples(root = 'сrop_images')

model_name = 'model_loss_0.00005.pth'
model = CRNN(imgH=IMG_HEIGHT, nc=1, nclass=num_classes, nh=256).to(device)
model.load_state_dict(torch.load(model_name))

model.eval()

count_good = 0
count = 0
with torch.no_grad():
    for img_path, num  in samples:
        img = Image.open(img_path).convert("L")
        img_tensor = transform(img).unsqueeze(0).to(device)
        preds = model(img_tensor)  
        res = decode(preds)
        count += 1
        if res[0] == num:
            count_good += 1
        
        # print(int(res[0]), num)
    print("accuracy: ",count_good/count)
