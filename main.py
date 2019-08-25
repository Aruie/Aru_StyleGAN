import torch
import torch.nn as nn
import torch.nn.functional as F

import os
import numpy as np
import PIL.Image as Image
import matplotlib.pyplot as plt

import model
from data_loader import data_loader
from tqdm import tqdm


def set_requires_grad(module, flag) :
    for p in module.parameters() :
        p.requires_grad = flag

# 학습부분 함수
def train(num_block, generator, discriminator, 
          batch_size, epochs, path_image) :

    d_losses = []
    g_losses = []

    # Progressive 학습 실행, 8x8 부터
    for step in range(2, num_block + 1) :

        # 에폭별실행
        #for epoch in tqdm(range(1, epochs[step] + 1)):
        for epoch in range(1, epochs[step] + 1):

            #데이터 로더 생성 (에포크당 최대 샘플 1000개)
            loader = data_loader(step, batch_size, path=path_image, num_workers=1)

            
            print(f'step = {step}, epoch = {epoch}')


            #생성된 로더로 이터레이션 실행
            for real_image in loader :

                # 가우시안 분포로 z 생성
                z = [torch.rand(100), torch.rand(100)]
                #z.append()
                #z.append(torch.rand(100))
                
                if torch.cuda.is_available() :
                    real_image = real_image.cuda()
                    z[0] = z[0].cuda()
                    z[1] = z[1].cuda()
 
            
                # Discriminator 학습
                discriminator.zero_grad()            
                set_requires_grad(generator, False)
                set_requires_grad(discriminator, True)

                # 기울기 계산 
                real_image.requires_grad = True
                real_predict = discriminator(real_image, step)
                real_predict = F.softplus(-real_predict).mean()
                real_predict.backward(retain_graph=True)

                # R1 패널티계산
                grad_real = torch.autograd.grad(outputs=real_predict.sum(), inputs=real_image, create_graph=True)[0]
                grad_penalty = (grad_real.view(grad_real.size(0), -1).norm(2, dim=1)**2).mean()
                grad_penalty = 10 / 2 * grad_penalty
                grad_penalty.backward()


                # Loss 계산
                fake_image = generator(z[0], step)
                fake_predict = discriminator(fake_image, step)
                
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

                fake_image = generator(z[0], step)
                fake_predict = discriminator(fake_image, step)
                fake_predict = F.softplus(-fake_predict).mean()
                fake_predict.backward()
                
                # 가중치 업데이트
                g_optim = torch.optim.Adam(generator.parameters(), lr=0.001)
                g_optim.step()

                g_losses.append(fake_predict.item())

    return d_losses, g_losses

def save_model(file_model, generator, discriminator ) :
    torch.save({
        'generator' : generator.state_dict(),
        'discriminator' : discriminator.state_dict()
        }, file_model)

def load_model(file_model) :
    model_dict = torch.load(file_model)
    return model_dict



# 메인함수
def main(num_block, epochs_list, batch_size, is_train, is_continue, is_save) :

    #####################################
    # 환경변수 지정, 추후 parser로 변환 예정
    #####################################
    
    #path_image = os.path.join(os.getcwd(), 'train_image/')
    path_image = os.path.join(os.getcwd(), '../datasets/DogData/')
    path_model = os.path.join(os.getcwd(), 'save_model/')
    print(f' Path of Image : {path_image}')

    model_name = 'model.pth'

    # 블록이 하나 이하일시 종료
    if(num_block <= 1) :
        print('Not enough block, Terminated')
        return 

    # 원하는 갯수의 블록을 가진 Generator 와 Discriminator 생성 할까 했는데 무조건 맥스로 생성
    generator = model.Generator(batch_size, 9)
    discriminator = model.Discriminator(9)

    if torch.cuda.is_available() == True : 
        generator = generator.cuda()
        discriminator = discriminator.cuda()
    
    if is_continue :  
        file_model = os.path.join(path_model, model_name)

        if os.path.exists(file_model) :  
            model_dict = load_model(file_model)
            generator.load_state_dict(model_dict['generator'])
            discriminator.load_state_dict(model_dict['discriminator'])
    
    # 학습 시작
    if is_train :
        train(num_block, generator, discriminator,
                 batch_size, epochs_list, path_image)
        print(f'Train End')

    if is_save :
        if os.path.exists(path_model) == False :
            os.mkdir(path_model)
        file_model = os.path.join(path_model, model_name)
        save_model(file_model, generator, discriminator)

    for i in range(5) :
        ###############################33
        # 임시 생성 테스트용
        z = torch.rand(100)
        if torch.cuda.is_available() :
            z = z.cuda()

        image = generator(z, num_block).cpu().detach().numpy()[0]
        image = image.transpose((1,2,0))
        img = Image.fromarray(np.uint8(image*255))
        img.save(os.path.join('save_image/',f'save{i}.png'), format='png')

    
    
    
# 테스트용
if __name__ == "__main__" :

    num_block = 5
    epochs_list = [-1, -1, 50, 50, 50, 50, 50, 0, 0, 0, 0]
    batch_size = 4

    main(num_block, epochs_list, batch_size, 
        is_train = True, is_continue = False, is_save = True)
    
    print('End of main')


