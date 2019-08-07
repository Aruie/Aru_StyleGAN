# A Style-Based Generator Architecture for Generative Adversarial Network

Nvidia에서 나온 논문인데 솔직히 혁신적이었다.
내가 GAN에 뛰어들게 만든 장본인... 꼭한번 구현해보고 싶었는데
마침 스터디도 하게 되었겠다 바로 시작!

일단 기존모델과의 차별점이라면

PGGAN을 베이스로하는 점차 커지는 네트워크를 이용하여 학습

Latent Space가 단순한 벡터가 아닌 z -> w로 만드는 맵핑 네트워크 구현

그리고 Style Mixing 을 사용해 Latent Space에 대한 의존성 삭제

nearest-neighbor 를 bilinear로 대체

