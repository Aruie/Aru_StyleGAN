import torch
from torch import nn
from torch.nn import functional as F


# 사용 채널들 리스트 
# 추후 제거 고민중 굳이 파이썬에서도 제거할 필요가 있을까
# list        0   1   2   3   4   5   6   7   8    9       
CHANNELS = [512,512,512,512,512,256,128, 64, 32,  16]
PIXELS =   [  0,  4,  8, 16, 32, 64,128,256,512,1024]
NOISE_PROB = [0,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1, 0.1]
EPSILON = 1e-8
Z_SIZE = 100
MAPPING_UINT = 512



# 생성기 정의
class Generator(nn.Module) :
    def __init__(self, batch_size, block_count = 9) :
        super(Generator, self).__init__()
        self.mapping = MappingNet()
        self.block = nn.ModuleDict()
        self.to_RGB = nn.ModuleDict()

        self.base = torch.randn(batch_size, CHANNELS[1], PIXELS[1], PIXELS[1])

        for i in range(block_count) :

            self.block[str(i+1)] = GBlock(i+1)
            self.to_RGB[str(i+1)] = ToRGB(i+1)


    def forward(self, z, step):

        x = self.base
        w = self.mapping(z)

        # 스텝 수만큼 레이어 반복
        for i in range(step) :
            x = self.block[str(i+1)](x, w, NOISE_PROB[i+1])

        #######################
        # 스무스한 변화를 위한 알파 적용 구현 필요
        #######################

        # 3채널로 변경
        x = self.to_RGB[str(i+1)](x)


        return x
        
            
# 생성기 내부 반복블럭 정의, step별 생성가능
class GBlock(nn.Module) :
    def __init__(self, step) :
        super(GBlock, self).__init__()

        self.step = step
        self.pixel = PIXELS[self.step]
        self.prev_channel = CHANNELS[self.step - 1]
        self.channel = CHANNELS[self.step]

        self.conv0 = nn.Conv2d(self.prev_channel, self.channel, 3, padding = 1)
        self.conv1 = nn.Conv2d(self.channel, self.channel, 3, padding = 1)

        # 현재 스텝에서 사용할 레이어의 크기 미리 저장, 계산 최소화
        self.layer_shape = [-1, 2, self.channel, 1, 1]
        self.noise_shape = [1, self.channel, self.pixel, self.pixel]

        # StyleMixing을 위해 W 에서 a, b 로 맵핑
        layer_size = 2 * self.channel
        self.style1 = nn.Linear(MAPPING_UINT, layer_size)
        self.style2 = nn.Linear(MAPPING_UINT, layer_size)

        # 그냥 업샘플
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')


    def forward(self, x, w, noise_prob) :

        # 업샘플 및 컨볼류션, 첫블록에서는 사용하지않음
        if self.step != 1 :
            x = self.upsample(x)
            x = self.conv0(x)

        ################
        # 노이즈 추가 - 추후 방식 변경
        ################
        noise = torch.randn(self.noise_shape)
        x = x + noise * noise_prob

        # 피쳐당 노말라이즈 실행, 배치당이 아님
        x = x - torch.mean(x, dim=(2,3), keepdim=True)
        p = torch.rsqrt(torch.mean(x**2, dim=(2,3), keepdim=True) + EPSILON) 
        x = torch.mul(p,x)

        # 생성된 스타일로 ax + b Pixelwise 연산
        style = self.style1(w)
        style = style.view(self.layer_shape)
        x = x * style[:,0] + style[:,1]

        # 위 과정 반복, 모듈화 시킬까 고민중
        x = self.conv1(x)

        noise = torch.randn(self.noise_shape)
        x = x + noise * noise_prob

        x = x - torch.mean(x, dim=(2,3), keepdim=True)
        p = torch.rsqrt(torch.mean(x**2, dim=(2,3), keepdim=True) + EPSILON) 
        x = torch.mul(p,x)

        style = self.style2(w)
        x = x * style[0] + style[1]

        return x

# 다채널 데이터를 3채널로 변경
class ToRGB(nn.Module) :
    def __init__(self, step) :
        super(ToRGB, self).__init__()
        self.conv = nn.Conv2d(CHANNELS[step] ,3, 1)

    def forward(self, x):
        return self.conv(x)

# 3채널 데이터를 레이어에 필요한 채널수로 변경
class FromRGB(nn.Module) :
    def __init__(self, step) :
        super(FromRGB, self).__init__()
        self.conv = nn.Conv2d(3, CHANNELS[step], 1)

    def forward(self, x) :
        return self.conv(x)
         
# Discriminator 정의
class Discriminator(nn.Module) :
    def __init__(self, block_count = 9) :
        super(Discriminator, self).__init__()
        self.block = nn.ModuleDict()
        self.from_RGB = nn.ModuleDict()

        for i in range(block_count, 0, -1) :
            self.from_RGB[str(i)] = FromRGB(i)
            self.block[str(i)] = DBlock(i)


    def forward(self, x, step) :

        #######################
        # 스무스한 변화를 위한 알파 적용 구현 필요
        #######################

        # 다채널 데이터로 변경, 전체 스텝에서 1회만 필요
        x = self.from_RGB[str(step)](x)

        # 블록 반복 실행
        for i in range(step, 0, -1) :
            x = self.block[str(i)](x)

        return x


class DBlock(nn.Module):
    def __init__(self, step):
        super(DBlock, self).__init__()

        self.step = step
        self.pixel = PIXELS[self.step]
        self.channel = CHANNELS[self.step]
        self.next_channel = CHANNELS[self.step - 1]

        self.conv1 = nn.Conv2d(self.channel, self.channel, 3, padding=1)
        self.leaky1 = nn.LeakyReLU(0.2)

        if self.step != 1 :
            self.conv2 = nn.Conv2d(self.channel, self.next_channel, 3, padding=1) 
            self.leaky2 = nn.LeakyReLU(0.2)
            self.avgpool = nn.AvgPool2d(2)

        else :
            self.conv2 = nn.Conv2d(self.channel, self.channel, 4, padding=0)
            self.leaky2 = nn.LeakyReLU(0.2)
            self.fc = nn.Linear(self.next_channel, 1)
        
    def forward(self, x) :
        
        x = self.conv1(x)
        x = self.leaky1(x)
        x = self.conv2(x)
        x = self.leaky2(x)

        if self.step != 1 :
            x = self.avgpool(x)

        else :
            ################################
            # minibatch standard deviation 구현해야됨
            ################################
            x = x.view(x.shape[0], -1)
            x = self.fc(x)

        return x
            

# Latent space 맵핑 네트워크 z > w
class MappingNet(nn.Module) :
    def __init__(self) :
        super(MappingNet, self).__init__()

        self.dense = nn.ModuleList([nn.Linear(Z_SIZE, MAPPING_UINT)])

        for i in range(7) :
            self.dense.append(nn.Linear(MAPPING_UINT, MAPPING_UINT))

    def forward(self, x) :
        # 추후 조절을 위해 입력을 따로 받음
        # 입력은 (100,1) 의 Unifrom-Dist 벡터
        for i in range(8) :
            x = self.dense[i](x)
            x = nn.ReLU()(x)

        return x

# 테스트
if __name__ == "__main__" :
    z = torch.rand(100)
 
    g = Generator(2)
    d = Discriminator()

    step = 3

    y = g(z, step)
    print(y.shape)

    z = d(y, step)
    print(z.shape)


