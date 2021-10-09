import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from PIL import Image
import numpy as np
import glob
import os

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    print("check cuda available!!")

#---------------------
# setting & constant
#---------------------
data_PATH = 'images/*.*'
label_PATH = None

batch_size = 4
epochs = 4

#---------------------
# define model
#---------------------
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(1000, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 128)
        self.fc5 = nn.Linear(128, 7)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.softmax(self.fc5(x))
        return x.view(-1,7)

#---------------------
# define DataLoader
#---------------------
class DataLoaders():
    def __init__(self, img_dir, transform=None):
        self.file_list = sorted(glob.glob(img_dir))
        self.transform = transform

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img = Image.open(self.file_list[idx])
        img = self.transform(img)
        label = [0]*7; label[int(self.file_list[idx][7:-4])%7] = 1
        label = np.float32(label)
        return img, label

# Image에 옵션 주기(데이터 변환)
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.Resize((512,512)),
    transforms.ColorJitter(), # 빼도 상관은 없음 
    transforms.ToTensor(), # 반드시 ToTensor 다음에 Normalize
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
])

#---------------------
# create model
#----------------------
model = models.resnet50(pretrained=True)
score_model = Model()

model.to(device)
model.eval()
score_model.to(device)

#---------------------
# create DataLoader
#---------------------
dataset = DataLoaders(data_PATH,transform)
train_loader = torch.utils.data.DataLoader(dataset=dataset,batch_size=batch_size,shuffle=True, drop_last=True) # drop_last : data % batch size 버림


import torch.optim as optim

criterion = nn.MSELoss()
optimizer = optim.Adam(score_model.parameters(), lr=0.001)
#---------------------
# train code
#---------------------
os.makedirs("result", exist_ok=True)
running_loss = 0.0
for epoch in range(epochs):  # loop over the dataset multiple times
    for i, data in enumerate(train_loader,0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data


        inputs = inputs.to(device)
        labels = labels.to(device) # label의 값을 -1이 없게 해주는 것이 좋음

        outputs = model(inputs)
        outputs = score_model(outputs)

        loss = criterion(outputs,labels)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # print statistics
        running_loss += loss.item()
        if i % 1000 == 0:    # 처음에 if내부 코드가 정상인지 체크
            print('[%d, %5d] loss: %.5f\n' %
                  (epoch, i, running_loss / 1000))
            running_loss = 0.0
            
            print(f'labels : \n{np.round(labels.cpu().squeeze().numpy())}')
            print(f'prediect : \n{np.around(np.float32(outputs.detach().cpu().squeeze()),3)}')
    PATH = f'./light_score_{epoch}.pt' 
    torch.save(score_model.state_dict(), PATH)        

print('Finished Training')

