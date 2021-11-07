"""
1. list_attr_celeba.txt 에서 원하는 attribute만 남겨놓고 저장하기(img : img_align_celeba, label:list_attr_celeba.txt)
2. data loader 만들기
3. torchvision에서 resnet50(논문에서 resnet50을 썼음) model을 가져오기(pretrained?)
4. resnet50의 마지막 layer을 attribute 개수(4)에 맞게 FCL 추가하기
5. 학습 코드 짜기
"""

import torch
# from torch._C import float32
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models

import numpy as np
import pandas as pd
import PIL
import PIL.Image
import glob
import os

#---------------------
# setting & constant
#---------------------
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print('Device:', device)
# print('Current cuda device:', torch.cuda.current_device())
# print('Count of using GPUs:', torch.cuda.device_count())

batch_size = 1
data_PATH = 'images/*.*'
# label_PATH = 'dataset/list_attr_celeba_3.txt'
#---------------------
# define DataLoader
#---------------------
class DataLoaders():
    def __init__(self, img_dir, transform=None):
        self.file_list = sorted(glob.glob(img_dir))
        # self.labels = np.loadtxt(label_dir,delimiter=' ') # 수정 필요
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img = PIL.Image.open(self.file_list[idx])
        img = self.transform(img)
        label = [0,0]
        label[int(self.file_list[idx].split('/')[1][0])] = 1
        label = np.array(label,dtype=float) # 수정 필요
        return img, label

transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])
#---------------------
# create DataLoader
#---------------------
dataset = DataLoaders(data_PATH,transform)
train_loader = torch.utils.data.DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True)

#-------------------------------------------
# Load pretrained Model & add Last Layer
#-------------------------------------------

class filter(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1000, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 2)

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.tanh(self.fc2(x))
        x = F.tanh(self.fc3(x))
        x = F.softmax(self.fc4(x),dim=-1)

        return x

model = models.resnet50(pretrained=True)
filter_model = filter()
filter_model.to(device)

# num_ftrs = model.fc.in_features
# model.fc = nn.Linear(num_ftrs,6)
model.to(device)
model.eval()

criterion = nn.BCELoss()
optimizer = optim.Adam(filter_model.parameters(), lr=0.001)

#-------------------------------------------
# train code
#-------------------------------------------
for epoch in range(6):  # loop over the dataset multiple times
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        
        outputs = model(inputs)
        outputs = filter_model(outputs.to(device))

        #  print(labels.re)
        loss = criterion(torch.tensor(outputs,dtype=float), labels.detach())
        loss.requires_grad_(True).backward()
        optimizer.step()
        optimizer.zero_grad()
        # import pdb;pdb.set_trace()
        running_loss += loss.item()
        if i % 500 == 0:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %(epoch + 1, i + 1, running_loss / 500))
            running_loss = 0.0

print('Finished Training')

PATH = './face_filter.pth'
torch.save(model.state_dict(), PATH)

# CUDA_VISIBLE_DEVICES=1 python main.py