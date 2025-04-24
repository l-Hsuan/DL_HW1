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
epochs = 30
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
log_dir = "/ssd1/hsuan/DL1/logs"
os.makedirs(log_dir, exist_ok=True)

# 使用時間戳記命名 log 檔案
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
log_filename = os.path.join(log_dir, f"test_log_{timestamp}.txt")

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

        # 這是 K 組「專家卷積核」
        self.weight_bank = nn.Parameter(torch.randn(K, out_channels, in_channels_max, kernel_size, kernel_size))

        # Attention score 生成器：使用 GAP + FC 層
        self.attn_pool = nn.AdaptiveAvgPool2d(1)
        self.attn_fc = nn.Sequential(
            nn.Linear(in_channels_max, 64),
            nn.ReLU(),
            nn.Linear(64, K),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        B, C, H, W = x.shape
        x_pad = F.pad(x, (0, 0, 0, 0, 0, self.in_channels_max - C))  # Zero padding to match max channels
        pooled = self.attn_pool(x_pad).view(B, -1)
        attn = self.attn_fc(pooled)  # Shape: (B, K)

        # Weighted combination of expert kernels
        weight = torch.einsum('bk,kocij->bocij', attn, self.weight_bank[:, :, :C, :, :])

        # Reshape and apply convolution per sample
        outputs = []
        for i in range(B):
            out = F.conv2d(x[i:i+1], weight[i], padding=self.kernel_size//2)
            outputs.append(out)
        return torch.cat(outputs, dim=0)

def select_channels(image, channels='RGB'):
    if channels == 'R':
        return image[0, :, :].unsqueeze(0)
    elif channels == 'G':
        return image[1, :, :].unsqueeze(0)
    elif channels == 'B':
        return image[2, :, :].unsqueeze(0)
    elif channels == 'RG':
        return image[0:2, :, :]
    elif channels == 'RB':
        return torch.stack([image[0, :, :], image[2, :, :]], dim=0)
    elif channels == 'GB':
        return image[1:, :, :]
    elif channels == 'RGB':
        return image

def rgb_dataloader(set_channel = 'RGB'):
    test_augmentations = transforms.Compose([
        transforms.Resize(96),
        transforms.CenterCrop(96),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.Lambda(lambda x: select_channels(x, set_channel))  # Change 'RGB' to other combinations as needed
    ])

    test_dataset = ImageNetDataset(data_file='/ssd1/hsuan/DL1/test.txt', root_dir='/ssd1/hsuan/DL1/', transform=test_augmentations)
    test_loader = DataLoader(test_dataset, batch_size=5, shuffle=False)
    return test_loader

def test_baseline(model):
    model.eval()
    correct = 0
    test_loader_len = len(test_loader.dataset)
    test_loader_iter = tqdm(test_loader, total=len(test_loader), desc="Testing")
    
    all_preds = []
    all_targets = []
    
    start_time = time.time()
    
    with torch.no_grad():
        for data, target in test_loader_iter:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            
            test_loader_iter.set_postfix(accuracy=100. * correct.item() / test_loader_len)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    accuracy = 100. * correct / test_loader_len
    
    # 計算 Precision, Recall 和 F1-score
    precision = 100. * precision_score(all_targets, all_preds, average='macro')
    recall = 100. * recall_score(all_targets, all_preds, average='macro')
    f1 = 100. * f1_score(all_targets, all_preds, average='macro')
    
    # 計算 FLOPS
    flops = torchprofile.profile_macs(model, torch.randn(1, *data.shape[1:]).to(device))
    
    return accuracy.item(), precision, recall, f1, flops, elapsed_time

def test(model, rgb_set):
    model.eval()
    correct = 0
    test_loader = rgb_dataloader(set_channel = rgb_set)
    test_loader_len = len(test_loader.dataset)
    test_loader_iter = tqdm(test_loader, total=len(test_loader), desc="Testing")
    
    all_preds = []
    all_targets = []
    
    start_time = time.time()
    
    with torch.no_grad():
        for data, target in test_loader_iter:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
            
            test_loader_iter.set_postfix(accuracy=100. * correct.item() / test_loader_len)
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    accuracy = 100. * correct / test_loader_len
    
    # 計算 Precision, Recall 和 F1-score
    precision = 100. * precision_score(all_targets, all_preds, average='macro')
    recall = 100. * recall_score(all_targets, all_preds, average='macro')
    f1 = 100. * f1_score(all_targets, all_preds, average='macro')
    
    # 計算 FLOPS
    flops = torchprofile.profile_macs(model, torch.randn(1, *data.shape[1:]).to(device))
    
    return accuracy.item(), precision, recall, f1, flops, elapsed_time


baseline = models.resnet34(pretrained=False)
baseline.fc = torch.nn.Linear(baseline.fc.in_features, 50)
baseline.load_state_dict(torch.load("/ssd1/hsuan/DL1/checkpoints/best_baseline.pt"))
baseline.to(device)
baseline.eval()
test_augmentations = transforms.Compose([
    transforms.Resize(96),
    transforms.CenterCrop(96),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
test_dataset = ImageNetDataset(data_file='/ssd1/hsuan/DL1/test.txt', root_dir='/ssd1/hsuan/DL1/', transform=test_augmentations)
test_loader = DataLoader(test_dataset, batch_size=5, shuffle=False)
resnet_acc, resnet_precision, resnet_recall, resnet_f1, resnet_flops, resnet_elapsed_time = test_baseline(baseline)
print(f"Accuracy: {resnet_acc:.2f}%, Precision: {resnet_precision:.2f}%, Recall: {resnet_recall:.2f}%, F1 Score: {resnet_f1:.2f}%, FLOPS: {resnet_flops:d}, Elapsed Time: {resnet_elapsed_time:.2f} seconds")



dynamic = models.resnet34(pretrained=False)
dynamic.conv1 = DynamicConv2d(in_channels_max=3, out_channels=64, kernel_size=3, K=4)
dynamic.fc = torch.nn.Linear(dynamic.fc.in_features, 50)
dynamic.load_state_dict(torch.load("/ssd1/hsuan/DL1/checkpoints/best_dynamic.pt"))
dynamic.to(device)
dynamic.eval()

rgb_list = ["RGB", "RG", "RB", "GB", "R", "G", "B"]
for rgb in rgb_list:
    dynamic_acc, dynamic_precision, dynamic_recall, dynamic_f1, dynamic_flops, dynamic_elapsed_time = test(dynamic,rgb)
    print(f"dynamic RGB Set: {rgb:s}, Accuracy: {dynamic_acc:.2f}%, Precision: {dynamic_precision:.2f}%, Recall: {dynamic_recall:.2f}%, F1 Score: {dynamic_f1:.2f}%, FLOPS: {dynamic_flops:d}, Elapsed Time: {dynamic_elapsed_time:.2f} seconds")
