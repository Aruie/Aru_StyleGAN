# A Style-Based Generator Architecture for Generative Adversarial Network

Nvidia에서 나온 논문인데 너무 감동받았다.
내가 GAN에 뛰어들게 만든 장본인... 꼭한번 구현해보고 싶었는데
마침 스터디도 하게 되었겠다 바로 시작!


참고자료는  
SRGAN 논문   
PGGAN 논문   
그리고 아래 깃허브 참조  
https://github.com/SiskonEmilia/StyleGAN-PyTorch/



## StyleGAN 모델의 특징

PGGAN을 베이스로하는 점차 커지는 네트워크를 이용하여 학습

Latent Space가 단순한 벡터가 아닌 z -> w로 만드는 맵핑 네트워크 구현

그리고 Style Mixing 을 사용해 Latent Space에 대한 의존성 삭제? 감소?

Noise 조절을 이용한 미세변화 및 텍스쳐 디테일 상승



Loss 부분은 아직 정확히 이해하지 못하여 그냥 가져다가 썻다 ㅠㅠ


## 파일설명

- model.py  
StyleGAN 모델이 정의된 파일.
Generator 및 Discriminator가 정의되어있음  

- data_loader.py  
데이터 불러오는 로더가 정의된 파일.  
지정폴더에 원본 이미지 파일만 있으면 자동으로 레이어별로 리사이즈됨.

- main.py  
실행 및 학습관련 함수가 정의된 파일.
Save 및 Load 구현.

- generate_image.py
학습완료된 모델로 데이터를 생성하는 함수가 정의된 파일.


