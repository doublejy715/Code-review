## InterfaceGAN의 일부를 가져옮
InterfaceGAN 중 generate_data를 대신한다. attribute scoring 네트워크 추가


### 1. data_preprocess.py
- Dataset : celebaHQ image, attribute record file
- use attribute : 성별, 미소, 머리, 나이, 안경, 메이크업
- 제공된 attribute record file은 40개의 attribute를 제공 그 중 6개 attribute를 가져오는 코드

### 2. main.py
- attribute score를 측정하는 network 생성 & 학습
- 논문에서는 ResNet50을 이용하였다고 언급
- ResNet50 model을 가져오고, linear layer를 추가하여 각 attribute의 점수가 나오도록 설정(linear layer를 학습한다.)

### 3. generate_data.py
- input : latent vector / output : attirbute score of latent vector
- latent vector를 이용해 StyleGAN에서 이미지를 생성한다. 생성된 이미지를 main.py의 네트워크에 넣어 attribute score를 계산한다.
- latent vector, attribute 별로 numpy file을 저장한다.

** 이후 InterfaceGAN의 train boundary.py에 input으로 이용**
