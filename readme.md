(Multiple GPU 문제 해결시까지 보류) 

# A Style-Based Generator Architecture for Generative Adversarial Network

Nvidia에서 나온 논문인데 너무 감동받았다.
내가 GAN에 뛰어들게 만든 장본인... 꼭한번 구현해보고 싶었는데
마침 스터디도 하게 되었겠다 바로 시작!

본격적으로 구현을 시작한 최초의 모델인데
왜 학습이 힘든걸 골랐을까 엄청 후회중....

참고자료는  
SRGAN [https://arxiv.org/abs/1812.04948]   
PGGAN [https://arxiv.org/abs/1710.10196](PGGAN)  

그리고 자세히 안나온 부분은 아래 깃허브 참조  
https://github.com/SiskonEmilia/StyleGAN-PyTorch/



# 진행중 및 추가필요작업
 - 모델 아키텍쳐
   - ~~GPU 기능 추가~~
   - ~~학습 샘플수 1000개 제한(학습속도 일정화, 더 많은 샘플을 쓰고싶으면 에폭을 늘리자)~~
   - ~~Save 및 Load 구현~~
   - ~~Minibatch stddev 구현~~
   - Smooth Resolution Change ( 해보고 비교할랬는데...)
   - Noise의 위치 수정 (학습하면서...)
   - Latent Space 2중화 (이것도 하면서..)

 - 기타 
    - Main.py Parser 구현 (이건 다되고 나중에해도 굳이...)
    - 전체 관리 Singleton 클래스 구현
    - ~~tqmd 적용~~ 원격에 잦은 오류 발생으로 패스

 - 그리고 학습하기.... 
    - 역시나 잘안된다 ㅠ 많은 에폭을 돌려보자














## StyleGAN 모델의 특징

PGGAN을 베이스로하는 점차 커지는 네트워크를 이용하여 학습

Latent Space가 단순한 벡터가 아닌 z -> w로 만드는 맵핑 네트워크 구현

그리고 Style Mixing 을 사용해 Latent Space에 대한 의존성 감소 및 자유로운 변형 

Noise 조절을 이용한 미세변화 및 텍스쳐 디테일 상승



Loss 는 그냥 VanilaGAN 에 사용했던 기본 로스에 정규화 추가


여기까지 인듯. 자세히는 추후에 완성하면서 ㅠㅠ


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
학습완료된 모델로 데이터를 생성하는 함수가 정의된 파일. 미구현.

- model_info.py
모델 정보를 가지고 있는 싱글톤 클래스.
미구현.

