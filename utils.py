import torchvision.transforms as transforms
import torch, csv, os

import numpy as np

import cv2

from const import IMG_HEIGHT

# Для цифр 0-9
alphabet = "0123456789"

# Для CTC нужно зарезервировать символ "blank"
num_classes = len(alphabet) + 1  # +1 для CTC blank

def text_to_labels(text):
    return [alphabet.index(c) for c in text]

# индексы -> строка
def labels_to_text(labels):
    return ''.join([alphabet[i] for i in labels if i < len(alphabet)])


def get_samples(root):
    
    csv_path = os.path.join(root, 'labels.csv')
    samples = []

    with open(csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            filename = os.path.join(root, row["filenames"])
            label = row["words"]
            samples.append((filename, label))

    return samples

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def resize_and_pad(image, target_height = IMG_HEIGHT, target_width = 40, pad_color=255):
    h, w = image.shape[:2]

    scale = target_height / h
    new_w = int(w * scale)
    resized = cv2.resize(image, (new_w, target_height))

    # Если ширина меньше target_width → добавим паддинг
    if new_w < target_width:
        pad = target_width - new_w
        left = pad // 2
        right = pad - left
        resized = cv2.copyMakeBorder(resized, 0, 0, left, right,
                                     borderType=cv2.BORDER_CONSTANT,
                                     value=pad_color)
    else:
        resized = cv2.resize(resized, (target_width, target_height))

    return resized


def decode(preds):
    # preds: (T, B, nclass)
    pred_labels = preds.softmax(2).argmax(2)  # (T, B)

    pred_labels = pred_labels.permute(1, 0)  # (B, T)

    results = []
    for labels in pred_labels:
        collapsed = []
        prev = -1
        for l in labels.cpu().numpy():
            if l != prev and l != len(alphabet):
                collapsed.append(l)
            prev = l
        text = labels_to_text(collapsed)
        results.append(text)

    return results


if __name__ == "__main__":
    os.makedirs('crop_imgs', exist_ok=True)
    for i in os.listdir('images'):
        if i.endswith('.png'):
            img = cv2.imread(os.path.join('images', i), cv2.IMREAD_GRAYSCALE)
            img_out = resize_and_pad(img, IMG_HEIGHT, 40)
            cv2.imwrite(os.path.join('crop_imgs', i), img_out)



