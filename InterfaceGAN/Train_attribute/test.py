import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from PIL import Image
import numpy as np
import os

device = torch.device("cuda:1")
# pt 파일이면 반드시 들고와야 하나?
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

#--------------------------------
# Load ResNet50 & attri score
#--------------------------------
resnet50 = models.resnet50(pretrained=True)
resnet50.to(device)
resnet50.eval()

score_model = score()
score_model.load_state_dict(torch.load('attribute_score.pt'))
score_model.to(device)
score_model.eval()

#optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
#model.load_state_dict(checkpoint['model'])
#optimizer.load_state_dict(checkpoint['optimizer'])

#--------------------------------
# TEST code
#--------------------------------
"""
for file_name in os.listdir('dataset/test') :
    img = Image.open('dataset/test/' + file_name)
    # img = self.transform(img)

    img = img.resize((224,224))
    trans = transforms.ToTensor()
    img = trans(img).float().to(device)
    outputs = resnet50(img.unsqueeze(0))
    outputs = score_model(outputs).detach().cpu().numpy()
    print(f"{file_name} : {outputs}")
"""

#
# 차원을 맞춰주자! model input : (b,c,w,h)
#--------------------------------
# attribute 
# 
# Black_Hair : 9 (1:black ~ 0:birght hair?)
# Eyeglasses : 16(0:no / 1:yes )
# Heavy_Makeup : 19(0:no / 1:yes)
# Male : 21(0:female / 1:male)
# Smiling : 32(0:no / 1:yes)
# Young : 40(0:old / 1:young)
#--------------------------------
black_hair, eyeglasses, makeup, male, smiling, young = [],[],[],[],[],[]

for file_name in os.listdir('dataset/test') :
    img = Image.open('dataset/test/' + file_name)
    
    # img = self.transform(img)

    img = img.resize((224,224))
    trans = transforms.ToTensor()
    img = trans(img).float().to(device)
    if img.shape[0]==4:
        continue
    outputs = resnet50(img.unsqueeze(0))
    outputs = score_model(outputs).detach().cpu().numpy()

    # record results 
    black_hair.append([file_name[:-4],outputs[0][0]])
    eyeglasses.append([file_name[:-4],outputs[0][1]])
    makeup.append([file_name[:-4],outputs[0][2]])
    male.append([file_name[:-4],outputs[0][3]])
    smiling.append([file_name[:-4],outputs[0][4]])
    young.append([file_name[:-4],outputs[0][5]])

# save results to npy file
np.save(f'attri_result/black_hair.npy',np.array(black_hair))
np.save(f'attri_result/eyeglasses.npy',np.array(eyeglasses))
np.save(f'attri_result/makeup.npy',np.array(makeup))
np.save(f'attri_result/male.npy',np.array(male))
np.save(f'attri_result/smiling.npy',np.array(smiling))
np.save(f'attri_result/young.npy',np.array(young))

# 일단 확률을 저장한다. 이후에 따로 sorting 하고 filter해야함
# file name(num) 0 or 1 형식으로 npy 파일을 attribute별로 만들어 줘야한다.
# 상위 2퍼센트 골라내고
# for not in 삭제 해서 골라낸다.
# 그리고 boundary 학습