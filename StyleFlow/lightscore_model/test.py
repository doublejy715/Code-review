import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from PIL import Image
import numpy as np
import os

device = torch.device("cuda:1")

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
        x = F.softmax(self.fc5(x),dim=1)
        return x.view(-1,7)

resnet50 = models.resnet50(pretrained=True)
resnet50.to(device)
resnet50.eval()

score_model = Model()
score_model.load_state_dict(torch.load('light_score_2.pt'))
score_model.to(device)
score_model.eval()

labels = []
for file_name in os.listdir('test/') :
    img = Image.open('test/' + file_name)

    img = img.resize((512,512))
    trans = transforms.ToTensor()
    img = trans(img).float().to(device)
    if img.shape[0]==4:
        continue
    
    outputs = resnet50(img.unsqueeze(0))
    outputs = score_model(outputs).detach().cpu()
    labels.append(outputs.squeeze(0).unsqueeze(-1).numpy())
    
np.save('result/lights.npy',np.float32(labels))