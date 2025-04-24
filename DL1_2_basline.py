# import torch
# import torch.nn as nn
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# from PIL import Image
# import os, time, torchprofile
# import torch.nn.functional as F
# from sklearn.metrics import precision_score, recall_score, f1_score
# from tqdm import tqdm
# import torch.optim as optim
# import torchvision.models as models
# import matplotlib.pyplot as plt

# import torch
# print(torch.version.cuda)
# device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
# num_classes = 50
# epochs = 50
# batch_size = 32
# # test_batch_size = 32
# lr = 0.1
# momentum = 0.9
# weight_decay = 1e-4
# # Define a custom dataset class
# class ImageNetDataset(Dataset):
#     def __init__(self, data_file, root_dir, transform=None):
#         self.root_dir = root_dir
#         self.transform = transform
#         # Read the data file
#         with open(data_file, 'r') as file:
#             self.data = file.readlines()

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         # Process each line
#         line = self.data[idx].strip()
#         img_path, label = line.split()
#         # Complete image path
#         img_path = os.path.join(self.root_dir, img_path)
#         image = Image.open(img_path).convert('RGB')
#         label = int(label)
        
#         if self.transform:
#             image = self.transform(image)

#         return image, label
    
# # Define transformations
# def get_transform(mode='train'):
#     if mode == 'train':
#         base_transforms = [
#             transforms.RandomResizedCrop(224),
#             transforms.RandomHorizontalFlip(p=0.5),
#             transforms.RandomVerticalFlip(p=0.1),
#             transforms.RandomPerspective(distortion_scale=0.2, p=0.2),
#             transforms.RandomRotation(15)
#         ]
#     else:
#         base_transforms = [
#             transforms.Resize(384),
#             transforms.CenterCrop(224)
#         ]
#     base_transforms += [
#         transforms.ToTensor(),
#         transforms.Normalize([0.485, 0.456, 0.406], 
#                              [0.229, 0.224, 0.225])
#     ]

#     return transforms.Compose(base_transforms)

# # DataLoader
# train_dataset = ImageNetDataset(data_file='/ssd4/hsuan/DL1/train.txt', root_dir='/ssd4/hsuan/DL1/', transform=get_transform())
# val_dataset = ImageNetDataset(data_file='/ssd4/hsuan/DL1/val.txt', root_dir='/ssd4/hsuan/DL1/', transform=get_transform('val'))
# test_dataset = ImageNetDataset(data_file='/ssd4/hsuan/DL1/test.txt', root_dir='/ssd4/hsuan/DL1/', transform=get_transform('test'))

# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# # input ResNet-34 
# model = models.resnet34(pretrained=False)
# model.fc = torch.nn.Linear(model.fc.in_features, num_classes)  
# model.to(device)






# # # define loss function and optimizer
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)


# # train ResNet-34 
# # training
# train_losses = []
# train_accuracies = []
# val_losses = []
# val_accuracies = []
# def train(epoch):
#     model.train()
#     running_loss = 0.0
#     correct = 0
#     total = 0

#     for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")):
#         inputs, targets = inputs.to(device), targets.to(device)

#         optimizer.zero_grad()
#         outputs = model(inputs)
#         loss = criterion(outputs, targets)
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item() * inputs.size(0)
#         _, predicted = outputs.max(1)
#         total += targets.size(0)
#         correct += predicted.eq(targets).sum().item()

#     epoch_loss = running_loss / total
#     epoch_acc = 100. * correct / total
#     train_losses.append(epoch_loss)
#     train_accuracies.append(epoch_acc)

#     print(f"[Train] Epoch {epoch} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}%")

# # validation
# def val():
#     model.eval()
#     running_loss = 0
#     correct = 0
#     total = 0

#     with torch.no_grad():
#         for inputs, targets in tqdm(val_loader, desc="[Validating]"):
#             inputs, targets = inputs.to(device), targets.to(device)
#             outputs = model(inputs)
#             loss = criterion(outputs, targets)

#             running_loss += loss.item() * inputs.size(0)
#             _, predicted = outputs.max(1)
#             total += targets.size(0)
#             correct += predicted.eq(targets).sum().item()

#     epoch_loss = running_loss / total
#     epoch_acc = 100. * correct / total
#     val_losses.append(epoch_loss)
#     val_accuracies.append(epoch_acc)

#     print(f"[Val]  Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}%")
#     return epoch_loss, epoch_acc

# best_val_acc = 0
# for epoch in range(1,epochs+1):
#     train(epoch)
#     val_loss, val_acc = val()
#     scheduler.step()

#     if val_acc > best_val_acc:
#         best_val_acc = val_acc
#         torch.save(model.state_dict(), '/ssd4/hsuan/DL1/checkpoints/resnet34_best_model.pth')
#         print(f"Saved new best model with acc: {val_acc:.2f}%")


# # plot loss and acc of train and val
# epochs_range = range(1, epochs + 1)

# plt.figure(figsize=(10, 5))
# plt.plot(epochs_range, train_losses, label='Train Loss')
# plt.plot(epochs_range, val_losses, label='Val Loss')
# plt.xlabel('Epoch')
# plt.ylabel('Loss')
# plt.title('Training and Validation Loss')
# plt.legend()
# plt.grid(True)
# plt.savefig('/ssd4/hsuan/DL1/checkpoints/resnet34_loss_curve.png')
# plt.show()

# plt.figure(figsize=(10, 5))
# plt.plot(epochs_range, train_accuracies, label='Train Accuracy')
# plt.plot(epochs_range, val_accuracies, label='Val Accuracy')
# plt.xlabel('Epoch')
# plt.ylabel('Accuracy (%)')
# plt.title('Training and Validation Accuracy')
# plt.legend()
# plt.grid(True)
# plt.savefig('/ssd4/hsuan/DL1/checkpoints/resnet34_accuracy_curve.png')
# plt.show()



# # eval test data
# model.load_state_dict(torch.load('/ssd4/hsuan/DL1/checkpoints/resnet34_best_model.pth',map_location='cpu'))
# model.to(device)
# model.eval()


# test_loss = 0
# correct = 0
# total = 0

# all_preds = []
# all_targets = []

# start_time = time.time()

# with torch.no_grad():
#     for data, target in test_loader_iter:
#         data, target = data.to(device), target.to(device)
#         output = model(data)
#         pred = output.data.max(1, keepdim=True)[1]
#         correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        
#         all_preds.extend(pred.cpu().numpy())
#         all_targets.extend(target.cpu().numpy())
        
#         test_loader_iter.set_postfix(accuracy=100. * correct.item() / test_loader_len)

# end_time = time.time()
# elapsed_time = end_time - start_time

# accuracy = 100. * correct / test_loader_len

# # 計算 Precision, Recall 和 F1-score
# precision = 100. * precision_score(all_targets, all_preds, average='macro')
# recall = 100. * recall_score(all_targets, all_preds, average='macro')
# f1 = 100. * f1_score(all_targets, all_preds, average='macro')

# # 計算 FLOPS
# flops = torchprofile.profile_macs(model, torch.randn(1, *data.shape[1:]).to(device))

# print(f"Accuracy: {accuracy.item():.2f}%, Precision: {precision:.2f}%, Recall: {recall:.2f}%, F1 Score: {f1:.2f}%, FLOPS: {flops:d}, Elapsed Time: {elapsed_time:.2f} seconds")



#------------------------------
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os, time, torchprofile
import torch.nn.functional as F
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm
import torch.optim as optim
import torchvision.models as models
import matplotlib.pyplot as plt
import logging

# Logging 設定
log_path = '/ssd4/hsuan/DL1/logs/resnet34_log.txt'
logging.basicConfig(
    filename=log_path,
    filemode='w',
    format='[%(asctime)s] %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
formatter = logging.Formatter('[%(asctime)s] %(levelname)s - %(message)s', '%Y-%m-%d %H:%M:%S')
console.setFormatter(formatter)
logging.getLogger('').addHandler(console)

print(torch.version.cuda)
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
num_classes = 50
epochs = 50
batch_size = 32
lr = 0.1
momentum = 0.9
weight_decay = 1e-4

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
        img_path = os.path.join(self.root_dir, img_path)
        image = Image.open(img_path).convert('RGB')
        label = int(label)
        if self.transform:
            image = self.transform(image)
        return image, label


def get_transform(mode='train'):
    if mode == 'train':
        base_transforms = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.RandomPerspective(distortion_scale=0.2, p=0.2),
            transforms.RandomRotation(15)
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

train_dataset = ImageNetDataset('/ssd4/hsuan/DL1/train.txt', '/ssd4/hsuan/DL1/', transform=get_transform())
val_dataset = ImageNetDataset('/ssd4/hsuan/DL1/val.txt', '/ssd4/hsuan/DL1/', transform=get_transform('val'))
test_dataset = ImageNetDataset('/ssd4/hsuan/DL1/test.txt', '/ssd4/hsuan/DL1/', transform=get_transform('test'))

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

model = models.resnet34(pretrained=False)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

train_losses, train_accuracies, val_losses, val_accuracies = [], [], [], []

def train(epoch):
    model.train()
    running_loss, correct, total = 0.0, 0, 0
    for batch_idx, (inputs, targets) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch}/{epochs}")):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    train_losses.append(epoch_loss)
    train_accuracies.append(epoch_acc)
    logging.info(f"[Train] Epoch {epoch} | Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}%")

def val():
    model.eval()
    running_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc="[Validating]"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100. * correct / total
    val_losses.append(epoch_loss)
    val_accuracies.append(epoch_acc)
    logging.info(f"[Val]  Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.2f}%")
    return epoch_loss, epoch_acc

best_val_acc = 0
for epoch in range(1, epochs + 1):
    train(epoch)
    val_loss, val_acc = val()
    scheduler.step()
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), '/ssd4/hsuan/DL1/checkpoints/resnet34_best_model.pth')
        logging.info(f"Saved new best model with acc: {val_acc:.2f}%")

# Plot loss and acc
epochs_range = range(1, epochs + 1)
plt.figure(figsize=(10, 5))
plt.plot(epochs_range, train_losses, label='Train Loss')
plt.plot(epochs_range, val_losses, label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.grid(True)
plt.savefig('/ssd4/hsuan/DL1/logs/resnet34_loss_curve.png')
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(epochs_range, train_accuracies, label='Train Accuracy')
plt.plot(epochs_range, val_accuracies, label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('/ssd4/hsuan/DL1/logs/resnet34_accuracy_curve.png')
plt.show()

# Test
model.load_state_dict(torch.load('/ssd4/hsuan/DL1/checkpoints/resnet34_best_model.pth', map_location='cpu'))
model.to(device)
model.eval()

all_preds, all_targets = [], []
test_loss, correct, total = 0, 0, 0
start_time = time.time()

with torch.no_grad():
    for data, target in tqdm(test_loader, desc="[Testing]"):
        data, target = data.to(device), target.to(device)
        output = model(data)
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        all_preds.extend(pred.cpu().numpy())
        all_targets.extend(target.cpu().numpy())

end_time = time.time()
elapsed_time = end_time - start_time
accuracy = 100. * correct / len(test_dataset)
precision = 100. * precision_score(all_targets, all_preds, average='macro')
recall = 100. * recall_score(all_targets, all_preds, average='macro')
f1 = 100. * f1_score(all_targets, all_preds, average='macro')
flops = torchprofile.profile_macs(model, torch.randn(1, 3, 224, 224).to(device))

logging.info(f"[Test Result] Accuracy: {accuracy:.2f}%, Precision: {precision:.2f}%, Recall: {recall:.2f}%, F1 Score: {f1:.2f}%, FLOPS: {flops:d}, Elapsed Time: {elapsed_time:.2f} seconds")

