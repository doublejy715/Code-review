import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as models

from skimage import io
import numpy as np
import pandas as pd
import PIL
from PIL import Image
import glob
import os

#---------------------
# setting & constant
#---------------------
device = torch.device("cuda")
batch_size = 4
img_w = 178
img_h = 218
data_PATH = 'dataset/img_align_celeba/*.*'
label_PATH = 'dataset/list_attr_celeba_3.txt'

#---------------------
# define DataLoader
#---------------------
class DataLoaders():
    def __init__(self, img_dir, label_dir, transform=None):
        self.file_list = sorted(glob.glob(img_dir))
        self.labels = np.loadtxt(label_dir,delimiter=' ')
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        label = np.array(self.labels[idx])
        # image 가져오고 transform 적용
        img = Image.open(self.file_list[idx])
        img = self.transform(img)
        return img, label


transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize((224,224)),
    transforms.ColorJitter(), # 빼도 상관은 없음 
    transforms.ToTensor() # 반드시 ToTensor 다음에 Normalize
])

# transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                         std=[0.229, 0.224, 0.225])

#---------------------
# create DataLoader
#---------------------
dataset = DataLoaders(data_PATH,label_PATH,transform)
train_loader = torch.utils.data.DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True, drop_last=True) # drop_last : data % batch size 버림

#-------------------------------------------
# Load pretrained Model & add Last Layer
#-------------------------------------------

class score(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1000, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 6)

        self.tanh = nn.Tanh()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x.view(-1,6) # reshape

model = models.resnet50(pretrained=True)
score_model = score()
# num_ftrs = model.fc.in_features
# model.fc = nn.Linear(num_ftrs,6)
model.to(device)
model.eval()
score_model.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(score_model.parameters(), lr=0.001)

#-------------------------------------------
# train code
#-------------------------------------------
running_loss = 0.0
for epoch in range(1):  # loop over the dataset multiple times
    for i, data in enumerate(train_loader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data

        inputs = inputs.float().to(device)
        labels = (labels.float().to(device)+1)/2 # output 0~1, 반올림시 -1 0 1  3type result -> no!

        outputs = model(inputs)
        outputs = score_model(outputs)

        # print(outputs.type(), labels.type())

        loss = criterion(outputs, labels)
        # loss = criterion(outputs.to(torch.long), labels.to(torch.long))
        loss.backward()
        optimizer.step()
        # zero the parameter gradients
        optimizer.zero_grad()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 0:    # 처음에 if내부 코드가 정상인지 체크
            print('[%d, %5d] loss: %.3f' %
                  (epoch, i, running_loss / 2000))
            running_loss = 0.0

            out = torchvision.utils.make_grid(torch.cat((inputs, inputs), 0), nrow=inputs.shape[0])
            out = transforms.ToPILImage()(out.cpu().squeeze().clamp(0, 1))
            os.makedirs("result", exist_ok=True)
            out.save(f"result/{epoch}_{i}_{np.round(labels.cpu().squeeze().numpy())}_{np.round(outputs.detach().cpu().squeeze().numpy())}.png") # detach // with torch no_grad // requires grad
print('Finished Training')

PATH = './attribute_score.pt' 
torch.save(score_model.state_dict(), PATH)