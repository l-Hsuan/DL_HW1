import os
import time
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import precision_score, recall_score, f1_score
from thop import profile

# ========== Logging Setup ==========
def setup_logging(log_path='/ssd1/hsuan/DL1/logs/DL2my_train_log2_notatten.txt'):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path)
        ]
    )

setup_logging()

# ========== Dataset ==========
class ImageNetDataset(Dataset):
    def __init__(self, data_file, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        with open(data_file, 'r') as file:
            self.data = file.readlines()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        line = self.data[idx].strip()
        img_path, label = line.split()
        image = Image.open(os.path.join(self.root_dir, img_path)).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, int(label)

def get_transform(mode='train'):
    if mode == 'train':
        transform_list = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15)
        ]
    else:
        transform_list = [
            transforms.Resize(256),
            transforms.CenterCrop(224)
        ]
    transform_list += [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]
    return transforms.Compose(transform_list)


import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2)  # 下採樣一半
        )

    def forward(self, x):
        return self.block(x)

class SelfAttentionBlock(nn.Module):
    def __init__(self, dim, heads=4):
        super(SelfAttentionBlock, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.view(B, C, -1).permute(0, 2, 1)  # [B, HW, C]
        attn_out, _ = self.attn(x_flat, x_flat, x_flat)
        x_out = self.norm(attn_out + x_flat)  # 殘差連接
        x_out = x_out.permute(0, 2, 1).view(B, C, H, W)  # reshape 回原圖
        return x_out

class CNN4ConvWithAttention(nn.Module):
    def __init__(self, num_classes):
        super(CNN4ConvWithAttention, self).__init__()
        self.conv1 = ConvBlock(3, 64)      # 84→42
        self.conv2 = ConvBlock(64, 128)    # 42→21
        self.conv3 = ConvBlock(128, 256)   # 21→10
        self.conv4 = ConvBlock(256, 256)   # 10→5

        # self.attn = SelfAttentionBlock(dim=256, heads=4)  # attention 作用在 5x5 特徵上

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # x = self.attn(x)       # 保持 shape 不變
        x = self.pool(x)       # [B, 256, 1, 1]
        x = x.view(x.size(0), -1)
        return self.classifier(x)




# ========== Utils ==========
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def count_flops(model, input_tensor):
    flops, params = profile(model, inputs=(input_tensor,))
    return flops, params

def measure_inference_time(model, input_tensor, device, repeat=50):
    model.eval()
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        _ = model(input_tensor)
        torch.cuda.synchronize()
        times = []
        for _ in range(repeat):
            start = time.time()
            _ = model(input_tensor)
            torch.cuda.synchronize()
            times.append(time.time() - start)
    return sum(times) / repeat

# ========== Train Function ==========
def train(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs):
    model.to(device)
    input_tensor = torch.randn(1, 3, 224, 224).to(device)
    flops, _ = count_flops(model, input_tensor)
    params = count_parameters(model)
    inf_time = measure_inference_time(model, input_tensor, device)

    logging.info(f"\nModel Profile:")
    logging.info(f"  Params: {params:,}")
    logging.info(f"  FLOPs: {flops / 1e9:.2f} GFLOPs")
    logging.info(f"  Inference Time: {inf_time*1000:.2f} ms/image")

    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        correct, total, running_loss = 0, 0, 0.0

        for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        train_acc = correct / total
        val_acc = evaluate(model, val_loader, device)
        logging.info(f"[Epoch {epoch+1}] Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), '/ssd1/hsuan/DL1/checkpoints/mybest_model_notatten.pth')
            logging.info(f"New Best Model Saved (Val Acc: {best_acc:.4f})")

        scheduler.step()

# ========== Eval ==========
def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return correct / total

# ========== Test ==========
def test(model, test_loader, device):
    model.eval()
    all_preds, all_labels = [], []
    total_time = 0.0
    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Testing"):
            inputs, labels = inputs.to(device), labels.to(device)
            start = time.time()
            outputs = model(inputs)
            torch.cuda.synchronize()
            end = time.time()

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            total_time += (end - start)

    acc = sum(p == t for p, t in zip(all_preds, all_labels)) / len(all_labels)
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    avg_time = total_time / len(all_labels)

    logging.info(f"\n Test Results:")
    logging.info(f"  Accuracy : {acc:.4f}")
    logging.info(f"  Precision: {precision:.4f}")
    logging.info(f"  Recall   : {recall:.4f}")
    logging.info(f"  F1-Score : {f1:.4f}")
    logging.info(f"  Avg Inference Time: {avg_time*1000:.2f} ms/image")

# ========== Main ==========
if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_classes = 50
    batch_size = 32
    epochs = 50

    train_dataset = ImageNetDataset('/ssd1/hsuan/DL1/train.txt', '/ssd1/hsuan/DL1/', get_transform('train'))
    val_dataset = ImageNetDataset('/ssd1/hsuan/DL1/val.txt', '/ssd1/hsuan/DL1/', get_transform('val'))
    test_dataset = ImageNetDataset('/ssd1/hsuan/DL1/test.txt', '/ssd1/hsuan/DL1/', get_transform('test'))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    model = CNN4ConvWithAttention(num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    train(model, train_loader, val_loader, criterion, optimizer, scheduler, device, epochs)

    model.load_state_dict(torch.load('/ssd1/hsuan/DL1/checkpoints/mybest_model_notatten.pth'))
    test(model, test_loader, device)
