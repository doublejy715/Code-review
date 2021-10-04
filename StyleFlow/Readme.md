# StyleFlow
## Code review
- Code review는 'StyleFlow-original'에 있습니다.
- UI관련 코드까지 살펴보진 않았습니다.
- StyleFLow/main.py에 latent code를 edit하는 방법이 있습니다. 해당 부분을 살펴보았습니다.

## StyleFlow 순서  
### 1. init_deep_model
#### 요약 : styleGAN, CNF chain model을 만든다.
input : opt / output : None  
관련 호출 함수  
> utils.py - class Build_model  
> module / flow.py - def cnf  

#### Process
1. StyleGAN2 .pkl 파일 load
    - G_ema 모델을 이용한다.(StyleFlow에서는 Gs라 칭함)

2. CNF block이 여러개 있는 chain model을 만든다.
    - Chain model  
        - 여러 [Normalize layer + CNF block] 을 가진 model  

3. chain model 파라미터 load
    - train_flow.py 에서 학습하여 저장한 .pt 파일을 load한다.  
    - train_flow.py 에서는 latent vector, attribute, lighting 정보를 가지고 flow를 학습 함.(정확히는 파악 x)

4. 만들어진 chain model은 'prior' 이라는 변수로 선언한다.

### 2. init_data_points
#### 요약 : TSNE에 따른 군집 분포에서 대표되는 좌표값에 점을 찍는다. 각 점에는 대응되는 w latent vector가 저장되어 있다.
input : None / output : None  
관련 호출 함수  
> NearestNeighbors - 가까운 점의 위치를 찾는 것 같은데 필요 없을 듯

#### Process
1. data 파일 안의 파일들을 load한다.
    - latent vectors
    - attributes
    - lights
    - TSNE

2. keep_indexes에 해당하는 latent vector, attributes, lights, TSNE 정보를 따로 저장한다.
    - keep_indexes는 정해져 있는 숫자들이 저장된 list이다.

3. 1024,1024 size의 map을 만들고 TSNE의 keep_indexes에 해당하는 좌표값에 점을 찍는다.
    - 해당 점을 누르면 latent vector를 선택하게 되고, 그에 해당하는 image를 StyleGAN에서 만들어 낸다.

### 3. update_GT_scene_image
#### 요약 : image editing을 하기 위한 여러 변수 정의
input : None / output : None  
#### Process
1. Image data 정의
    - 이미지의 w latent vector, attribute, light 정보를 가져온다.
    - attribute, light 정보는 따로 list를 만들어 저장한다.  
    (실제 StyleFlow에서는 init_data_points에서 선택된 점의 latent vector, attribute, light 정보를 가져옴)  

2. 다른 변수 정의
    - q_array : 현재 사진의 w latent vector를 저장한다.
    - array_source : 현재 사진의 attribute value를 저장
    - array_light : 현재 사진의 light value 저장
    - pre_lighting_distance : lighting value에 영향을 주기 위한 distance 값 저장
    - final_array_source : light + attribute 값을 cat으로 이어준다.
    - final_array_target : final_array_source 와 같음
    - fws : chain block function을 거친 output을 저장한다.(forwards)  
        정방향으로 chain block을 거친 경우, attribute를 조정할 수 있는 latent space로 vector가 mapping 된다.
        - input : q_array, final_array_source, zero_padding
        - function : chain block function
        - output : 아마 final_array_source로 인해서 다시 mapping 된 latent vector 값
    - GAN_image : 현재 w latent vector로 GAN을 통해 만들어진 Image

### 4. real_time_editing(attribute)
#### 요약 : image editing 하기 위해서 w latent vector 값을 수정한다. 그리고 수정된 w latent vector을 가지고 이미지 생성
#### Process
1. UI에서 silde bar의 값을 실제 latent vector를 조정하기 위한 값으로 바꿔준다.(좌표계 바꾸기)
2. attribute 값에다가 1번의 조정 값을 연산한다.(연산 과정은 사칙연산으로 이루어져 있음)
3. 위에서 정의한 fws(source image의 forwards latent vector)와 조정하고 싶은 attribute 정보를 chain block에 넣어준다. chain block은 역방향으로 흘러가게 된다.
    - 역방향으로 넣어주게 되면, StyleGAN2 latent space에 맞는 editing image의 latent vector를 얻을 수 있다.
4. 원래의 identity를 살리기 위해서 원래 사진의 w latent vector(q_array)의 일부 정보를 그대로 복사한다.
5. editing image의 정보를 백업한다.(이어서 editing하기 위함)
    - w_current, q_array, fws, GAN_image 변수에 백업
    - GAN_image에는 editing image가 저장된다.

### 5. real_time_lighting
4.real_time_editing(attribute) 와 비슷한 과정을 가지고 있다.

## Related code for editing
### In UI
- 상단
주로 UI좌측의 attribute, light index / value를 기록되어 있다.  
추가적으로 각 attribute value의 mix / min 값이 기록되어 있다.  
slide bar value -> real value 로 바꿔주는 함수가 있다.  
real value -> slide bar value 로 바꿔주는 함수가 있다.  
(물론 attribute, light 별로 함수가 정의되어 있다.)  

### uilts.py
- 모델을 불러오고 모델의 초기 설정을 진행합니다.

## Edit code
### 잘 안된 점
- UI와 StyleGAN을 동시에 돌리기 때문에 thread를 이용하였음. thread를 이용하기 위해서 1.x tenserflow를 이용함. base는 pytorch임
- 1.x tensorflow는 CUDA 10.0 까지 지원했음.(나는 CUDA 11.4임 재설치 귀찮)
- 나는 UI없이 돌리기를 원했고, pytorch를 이용함. -> tensorflow code 없애서 CUDA도 바꾸지 않도록 함.


### 수정 방향
- UI없이 원하는 attribute만 수정하고자 함

#### Input
- 따로 parser를 만들고, 한번에 여러 attribute를 수정할 수 있도록 함
- input의 경우를 여러가지로 둠 : seed number, edit원하는 이미지 input

#### Modify
- 논문에서 말한 DPR model(light score), microsoft face API을 추가 <- TSNE 군집을 이용하지 않기 위함
- StyleGAN2 에서 학습한 pkl file을 가지고 바로 이용할 수 있도록 수정(학습 최소화)
- CNF model input shape을 다르게 해줌(light score의 수가 7개 밖에 없음... 원래는 index 9개)