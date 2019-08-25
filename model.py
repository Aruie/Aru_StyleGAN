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

        self.base = nn.Parameter(torch.randn(batch_size, CHANNELS[1], PIXELS[1], PIXELS[1]))

        for i in range(1, block_count+1) :

            self.block[str(i)] = GBlock(i)
            self.to_RGB[str(i)] = ToRGB(i)


    def forward(self, z, step, alpha=0):

        x = self.base
        w = self.mapping(z)

        # 스텝 수만큼 레이어 반복
        for i in range(1, step+1) :
            x = self.block[str(i)](x, w, NOISE_PROB[i])

        x = self.to_RGB[str(i)](x)
        
        # 3채널로 변경
        return x    

        #######################
        # 스무스한 변화를 위한 알파 적용 구현 필요    - 보류
        #######################

        #for i in range(1, step) :
        #    x = self.block[str(i)](x, w, NOISE_PROB[i+1])
        #ori = nn.Upsample(scale_factor=2, mode='bilinear')(x)
        #new = self.block[str(step)](x, w, NOISE_PROB[i+1])

        #ori = self.to_RGB[str(step)](ori) * alpha
        #new = self.to_RGB[str(step)](new) * (1-alpha)
        #x = ori + new

        
        
            
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

        # noise 미리지정 
        self.noise1 = nn.Parameter(torch.randn(self.noise_shape))
        self.noise2 = nn.Parameter(torch.randn(self.noise_shape))

        # 그냥 업샘플
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

        # 리키렐루
        self.leaky1 = nn.LeakyReLU(0.2)
        self.leaky2 = nn.LeakyReLU(0.2)
          


    def forward(self, x, w, noise_prob) :

        # 업샘플 및 컨볼류션, 첫블록에서는 사용하지않음
        if self.step != 1 :
            x = self.upsample(x)
            x = self.conv0(x)
            x = self.leaky1(x)

        ################
        # 노이즈 추가 - 추후 방식 변경
        ################
        noise = self.noise1
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
        x = self.leaky2(x)

        noise = self.noise2
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

        self.leaky1 = nn.LeakyReLU(0.2)
        self.leaky2 = nn.LeakyReLU(0.2)
            

        self.stddev = MinibatchStandardDeviation()


        if self.step != 1 :
            self.conv1 = nn.Conv2d(self.channel, self.channel, 3, padding=1)
            self.conv2 = nn.Conv2d(self.channel, self.next_channel, 3, padding=1) 
            self.avgpool = nn.AvgPool2d(2)

        else :
            self.conv1 = nn.Conv2d(self.channel+1, self.channel, 3, padding=1)
            self.conv2 = nn.Conv2d(self.channel, self.channel, 4, padding=0)
            self.fc = nn.Linear(self.next_channel, 1)

        
        
    def forward(self, x) :

        if self.step == 1 :
            # minibatch standard deviation 구현
            x = self.stddev(x)

        x = self.conv1(x)
        x = self.leaky1(x)
        x = self.conv2(x)
        x = self.leaky2(x)

        if self.step != 1 :
            x = self.avgpool(x)

        else :

       

            x = x.view(x.shape[0], -1)
            x = self.fc(x)

        return x

class MinibatchStandardDeviation(nn.Module) :
    def __init__(self) :
        super(MinibatchStandardDeviation, self).__init__()

    def forward(self, x) :
        s = x.shape
        y = x - x.mean(dim=0, keepdim=True)
        y = (y**2).mean(0)
        y = torch.sqrt(y + EPSILON)
        y = y.mean()
        y = y.expand((s[0],1,s[2],s[3]))
        x = torch.cat([x, y], 1)
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
    z = torch.rand(100).cuda()
 
    g = Generator(9)
    g = g.cuda()
    d = Discriminator(9)
    d = d.cuda()
    step = 6


    y = g(z, step)
    print(y.shape)

    z = d(y, step)
    print(z.shape)



