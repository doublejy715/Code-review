# SeFa - Closed-Form Factorization of Latent Semantics in GANs

- .pkl file에서 sefa를 이용하고 싶어서 만든 버전이다.

- .pkl file은 styleGAN2 ada에서 만들어진 모델이다. 

## 수정한 부분
- .pkl file을 불러온다.
- semantic별로 폴더를 만들고 안에 png file을 저장한다.

## 어려웠던 점

- original version에서는 .pth file을 이용하였다.
- utils.py 에서 .pth file안에서 style weight 부분이 .pkl file에서는 어떻게 정의되는지 알아야 했다.

  `weight = generator.synthesis.__getattr__(layer_name).style.weight.T`

## 해결 방법
- model을 정의할 때 어떤 layer가 style.weight으로 바뀌었는지 확인하였다.
- 알아보기 위해 styleGAN2-ada github homepage에 들어가 .pkl 파일로 저장할 때 어떤 이름으로 바뀌는지 확인하였다.
- [ 참조 Link ](https://github.com/NVlabs/stylegan2-ada-pytorch/blob/main/legacy.py)
- 168line부터 바뀌는 이름이 보인다.
- 대응되는 부분을 코드에서 접근하여 original code와 동일한 weight에 접근할 수 있도록 하였다.

## 추가
- model을 idol data trained model로 테스트 해 보았다.