import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt

import model
from data_loader import data_loader




def set_requires_grad(module, flag) :
    for p in module.parameters() :
        p.requires_grad = flag

# 학습부분 함수
def train(num_block, generator, discriminator, 
          batch_size, epochs, path_image) :


    d_losses = []
    g_losses = []

    # 가우시안 분포로 z 생성
    z = torch.rand(100)

    # Progressive 학습 실행, 8x8 부터
    for i in range(2, num_block + 1) :

        # 에폭별실행, 추후 step별 에폭 조정 기능 추가
        for _ in range(epochs) :
            
            # 데이터로더 생성 및 데이터 가져오기
            # 로더를 에폭마다 생성하는데 step마다 생성하는것으로 변경 예정
            loader = iter(data_loader(i, batch_size, path=path_image, num_workers=0 ))
            real_image = next(loader)
            
            # Discriminator 학습
            discriminator.zero_grad()            
            set_requires_grad(generator, False)
            set_requires_grad(discriminator, True)

            # 기울기 계산 
            real_image.requires_grad = True
            real_predict = discriminator(real_image, i)
            real_predict = F.softplus(-real_predict).mean()
            real_predict.backward(retain_graph=True)

            # R1 패널티계산
            grad_real = torch.autograd.grad(outputs=real_predict.sum(), inputs=real_image, create_graph=True)[0]
            grad_penalty = (grad_real.view(grad_real.size(0), -1).norm(2, dim=1)**2).mean()
            grad_penalty = 10 / 2 * grad_penalty
            grad_penalty.backward()

            # Loss 계산
            fake_image = generator(z, i)
            fake_predict = discriminator(fake_image, i)
            
            fake_predict = F.softplus(fake_predict).mean()
            fake_predict.backward()
            
            d_losses.append((real_predict + fake_predict).item())

            # 가중치 업데이트
            d_optim = torch.optim.Adam(discriminator.parameters(), lr=0.001)
            d_optim.step()

            # 메모리 반환
            del fake_image, real_image, grad_penalty, grad_real

            # Generator 학습
            generator.zero_grad()
            set_requires_grad(discriminator, False)
            set_requires_grad(generator, True)

            fake_image = generator(z, i)
            fake_predict = discriminator(fake_image, i)
            fake_predict = F.softplus(-fake_predict).mean()
            fake_predict.backward()
            
            # 가중치 업데이트
            g_optim = torch.optim.Adam(generator.parameters(), lr=0.001)
            g_optim.step()

            g_losses.append(fake_predict.item())

    return d_losses, g_losses

# 메인함수
def main(num_block) :

    #####################################
    # 환경변수 지정, 추후 parser로 변환 예정
    #####################################
    num_block = num_block
    epochs = 5
    batch_size = 4
    path_image = os.path.join(os.getcwd(), 'train_image/')
    path_model = os.path.join(os.getcwd(), 'save_model/')
    print(f' Path of Image : {path_image}')

    # 블록이 하나 이하일시 종료
    if(num_block <= 1) :
        print('Not enough block, Terminated')
        return 

    # 원하는 갯수의 블록을 가진 Generator 와 Discriminator 생성
    generator = model.Generator(batch_size, num_block)
    discriminator = model.Discriminator(num_block)

    # 학습 시작
    train(num_block, generator, discriminator,
          batch_size, epochs, path_image)
    print(f'Train End')

    # 임시 생성
    z = torch.rand(100)
    image = generator(z, num_block).detach().numpy()[0]
    image = image.transpose((1,2,0))
    img = Image.fromarray(np.uint8(image*255))
    img.save('save.png', format='png')
    
    
# 테스트용
if __name__ == "__main__" :
    main(6)

    
    print('End of main')


