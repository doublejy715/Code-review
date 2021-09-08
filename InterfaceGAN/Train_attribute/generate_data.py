import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

import numpy as np
import os
import legacy
import dnnlib
import torch
import PIL

device = torch.device("cuda")

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

with dnnlib.util.open_url('idol_nohand_400.pkl') as f:
    G = legacy.load_network_pkl(f)['G_ema'].to(device) # type: ignore

face_pool = torch.nn.AdaptiveAvgPool2d((224, 224)).eval()

z_list,black_hair, eyeglasses, makeup, male, smiling, young = [],[],[],[],[],[],[]
count = 0
for seed in range(500000):
    print(seed)
    z = torch.from_numpy(np.random.RandomState(seed).randn(1, 512)).to(device) # seed -> numpy(1,512) -> G
    img = G(z, torch.zeros_like(z).to(device), truncation_psi=0.7, noise_mode='const') # img : image value -1 ~ 1
    img = face_pool((img+1)/2) # (img+1)/2 : image value 0 ~ 1

    # img_save = (img.permute(0, 2, 3, 1) * 255).clamp(0, 255).to(torch.uint8)
    # PIL.Image.fromarray(img_save[0].cpu().numpy(), 'RGB').save(f'generate/{str(seed).zfill(5)}.png')

    # G : -1 ~ 1 <->  resnet50 input : 0 ~ 1
    outputs = resnet50(img) 
    outputs = score_model(outputs).detach().cpu().numpy()
    # [[1 1 1 1 1]]
    # record results 
    black_hair.append([outputs[0][0]])
    eyeglasses.append([outputs[0][1]])
    makeup.append([outputs[0][2]])
    male.append([outputs[0][3]])
    smiling.append([outputs[0][4]])
    young.append([outputs[0][5]])

    z_list.append(z.cpu().numpy())
    count += 1

# save results to npy file
np.save('attri_result/black_hair_scores.npy',black_hair)
np.save('attri_result/eyeglasses_scores.npy',eyeglasses)
np.save('attri_result/makeup_scores.npy',makeup)
np.save('attri_result/male_scores.npy',male)
np.save('attri_result/smiling_scores.npy',smiling)
np.save('attri_result/young_scores.npy',young)
np.save('attri_result/z.npy',np.array(z_list).squeeze(1))
print('JOB END!')
