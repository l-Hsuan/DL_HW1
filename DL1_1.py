import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os, time, torchprofile,random
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm
import torch.optim as optim
import sys,datetime
import torchvision.models as models
os.environ.pop('LD_LIBRARY_PATH', None)

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# device = 'cpu'
num_classes = 50
epochs = 50
batch_size =  32
# test_batch_size = 32
lr = 0.1
momentum = 0.9
weight_decay = 1e-4


# 自訂 Logger 類別，同時輸出到 terminal 和檔案
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()  # 確保即時寫入

    def flush(self):
        pass

# 自動建立 logs 資料夾（如果還沒建立）
log_dir = "/ssd4/hsuan/DL1/logs"
os.makedirs(log_dir, exist_ok=True)

# 使用時間戳記命名 log 檔案
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = os.path.join(log_dir, f"train_log_{timestamp}.txt")

# 啟用 log 重導向
sys.stdout = Logger(log_filename)
print(f"[Log Start] Logging to: {log_filename}")

# Define a custom dataset class
class ImageNetDataset(Dataset):
    def __init__(self, data_file, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        # Read the data file
        with open(data_file, 'r') as file:
            self.data = file.readlines()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Process each line
        line = self.data[idx].strip()
        img_path, label = line.split()
        # Complete image path
        img_path = os.path.join(self.root_dir, img_path)
        image = Image.open(img_path).convert('RGB')
        label = int(label)
        
        if self.transform:
            image = self.transform(image)

        return image, label

class RandomChannelDropout:
    def __init__(self):
        self.channel_combinations = [
            (0, 1, 2),  # RGB
            (0, 1),     # RG
            (0, 2),     # RB
            (1, 2),     # GB
            (0,),       # R
            (1,),       # G
            (2,)        # B
        ]

    def __call__(self, img):
        channels = list(img.split())
        chosen_combination = random.choice(self.channel_combinations)
        combined_img = Image.merge('RGB', [channels[i] if i in chosen_combination else Image.new('L', img.size) for i in range(3)])
        return combined_img

# Define transformations
def get_transform(mode='train'):
    if mode == 'train':
        base_transforms = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.2),
            transforms.RandomRotation(15),
            RandomChannelDropout(),
        ]
    else:
        base_transforms = [
            transforms.Resize(384),
            transforms.CenterCrop(224)
        ]
    base_transforms += [
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], 
                             [0.229, 0.224, 0.225])
    ]

    return transforms.Compose(base_transforms)

class DynamicConv2d(nn.Module):
    def __init__(self, in_channels_max, out_channels, kernel_size, K=4):
        super().__init__()
        self.K = K  # Number of expert kernels
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.in_channels_max = in_channels_max

        # K groups of "expert" convolution kernels
        self.weight_bank = nn.Parameter(torch.randn(K, out_channels, in_channels_max, kernel_size, kernel_size))

        # Attention score generator: GAP + FC layers
        self.attn_pool = nn.AdaptiveAvgPool2d(1)
        self.attn_fc = nn.Sequential(
            nn.Linear(in_channels_max, 64),
            nn.ReLU(),
            nn.Linear(64, K),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        B, C, H, W = x.shape

        # Zero padding to match max channels
        if C < self.in_channels_max:
            x_pad = F.pad(x, (0, 0, 0, 0, 0, self.in_channels_max - C))  # Padding channels
        else:
            x_pad = x

        # Attention pooling: Global Average Pooling (GAP) + FC layers
        pooled = self.attn_pool(x_pad).view(B, -1)  # Shape: (B, C)
        attn = self.attn_fc(pooled)  # Shape: (B, K)

        # Weighted combination of expert kernels
        # Each B corresponds to a distribution of K expert kernels
        weight = torch.einsum('bk,kocij->bocij', attn, self.weight_bank[:, :, :C, :, :])  # Shape: (B, out_channels, C, kernel_size, kernel_size)

        # Perform convolution for each sample
        outputs = []
        for i in range(B):
            out = F.conv2d(x[i:i+1], weight[i], padding=self.kernel_size // 2)
            outputs.append(out)

        return torch.cat(outputs, dim=0)

def train(epoch,model, optimizer, train_loade):
    model.train()
    avg_loss = 0.
    train_acc = 0.
    train_loader_len = len(train_loader.dataset)
    train_loader_iter = tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Training Epoch #{epoch}")
    
    for batch_idx, (data, target) in train_loader_iter:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
        avg_loss += loss.item()
        pred = output.data.max(1, keepdim=True)[1]
        train_acc += pred.eq(target.data.view_as(pred)).cpu().sum()
        loss.backward()
        optimizer.step()
        train_loader_iter.set_postfix(loss=loss.item(), accuracy=100. * train_acc.item() / train_loader_len)
    
    print(f'Train Epoch: {epoch}, Loss: {avg_loss / len(train_loader):.6f}, Accuracy: {100. * train_acc / train_loader_len:.2f}%')

def val(epoch, model, val_loader):
    model.eval()
    test_loss = 0.
    correct = 0
    val_loader_iter = tqdm(val_loader, total=len(val_loader), desc=f"Validation Epoch #{epoch}")
    
    with torch.no_grad():
        for data, label in val_loader_iter:
            data, label = data.to(device), label.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, label, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(label.data.view_as(pred)).cpu().sum()
            val_loader_iter.set_postfix(loss=test_loss / len(val_loader.dataset), accuracy=100. * correct.item() / len(val_loader.dataset))
    
    test_loss /= len(val_loader.dataset)
    accuracy = 100. * correct / len(val_loader.dataset)
    print(f'Validation Set: Average Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%')

    return accuracy



# Paths
data_root = '/ssd4/hsuan/DL1/'

# Loaders
train_set = ImageNetDataset('/ssd4/hsuan/DL1/train.txt', data_root, get_transform('train'))
val_set = ImageNetDataset('/ssd4/hsuan/DL1/val.txt',data_root, get_transform('val'))
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
# 測試某個圖像
image, label = train_set[0]
print(image.shape)  # 應該返回 torch.Size([3, H, W])，而不是 PIL.Image
# Models

baseline = models.resnet34(pretrained=False)
baseline.fc = torch.nn.Linear(baseline.fc.in_features, num_classes)  
baseline.to(device)

K = 4
dynamic = models.resnet34(pretrained=False)
dynamic.conv1 = DynamicConv2d(in_channels_max=3, out_channels=64, kernel_size=3, K=K)
dynamic.fc = torch.nn.Linear(dynamic.fc.in_features, num_classes)  
dynamic.to(device)

# Optimizers & Loss
criterion = nn.CrossEntropyLoss()
optimizer_baseline = torch.optim.SGD(baseline.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
optimizer_dynamic = torch.optim.SGD(dynamic.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

# Train the models
best_val_acc_baseline = 0.
best_val_acc_dynamic = 0.

for epoch in range(epochs):
    # Train both models
    print('baseline')
    train(epoch + 1, baseline, optimizer_baseline, train_loader)
    print('dynamic')    
    train(epoch + 1, dynamic, optimizer_dynamic, train_loader)
    
    # Validate both models
    temp_acc_baseline = val(epoch + 1, baseline, val_loader)
    temp_acc_dynamic = val(epoch + 1, dynamic, val_loader)
    
    # Save the best model for SimpleCNN
    if temp_acc_baseline > best_val_acc_baseline:
        best_val_acc_baseline = temp_acc_baseline
        torch.save(baseline.state_dict(), '/ssd4/hsuan/DL1/checkpoints/best_baseline.pt')
        print(f'Best Baseline Accuracy: {best_val_acc_baseline:.2f}%')
    
    # Save the best model for DynamicCNN
    if temp_acc_dynamic > best_val_acc_dynamic:
        best_val_acc_dynamic = temp_acc_dynamic
        torch.save(dynamic.state_dict(), '/ssd4/hsuan/DL1/checkpoints/best_dynamic.pt')
        print(f'Best DynamicCNN Accuracy: {best_val_acc_dynamic:.2f}%')

print(f'Final Best Baseline Accuracy: {best_val_acc_baseline:.2f}%')
print(f'Final Best DynamicCNN Accuracy: {best_val_acc_dynamic:.2f}%')











