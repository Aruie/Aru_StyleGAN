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

        for i in range(step) :
            x = self.block[str(i+1)](x, w, NOISE_PROB[i+1])

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

        # 1블록에선 con1은 작동하지 않음
        self.conv0 = nn.Conv2d(self.prev_channel, self.channel, 3, padding = 1)
        self.conv1 = nn.Conv2d(self.channel, self.channel, 3, padding = 1)

        self.layer_shape = [-1, 2, self.channel, 1, 1]
        self.noise_shape = [1, self.channel, self.pixel, self.pixel]

        layer_size = 2 * self.channel
        self.style1 = nn.Linear(MAPPING_UINT, layer_size)
        self.style2 = nn.Linear(MAPPING_UINT, layer_size)

        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')


    def forward(self, x, w, noise_prob) :

        if self.step != 1 :
            x = self.upsample(x)
            x = self.conv0(x)


        # Add Noise
        noise = torch.randn(self.noise_shape)
        x = x + noise * noise_prob

        # Instance Normailize
        x = x - torch.mean(x, dim=(2,3), keepdim=True)
        p = torch.rsqrt(torch.mean(x**2, dim=(2,3), keepdim=True) + EPSILON) 
        x = torch.mul(p,x)

        # Style Mixing 
        style = self.style1(w)
        style = style.view(self.layer_shape)
        x = x * style[:,0] + style[:,1]

        # Convolution layer
        x = self.conv1(x)

        # Add Noise
        noise = torch.randn(self.noise_shape)
        x = x + noise * noise_prob

        # Instance Normailize
        x = x - torch.mean(x, dim=(2,3), keepdim=True)
        p = torch.rsqrt(torch.mean(x**2, dim=(2,3), keepdim=True) + EPSILON) 
        x = torch.mul(p,x)

        # Style Mixing 
        style = self.style2(w)
        x = x * style[0] + style[1]

        return x


class ToRGB(nn.Module) :
    def __init__(self, step) :
        super(ToRGB, self).__init__()
        self.conv = nn.Conv2d(CHANNELS[step] ,3, 1)

    def forward(self, x):
        return self.conv(x)

class FromRGB(nn.Module) :
    def __init__(self, step) :
        super(FromRGB, self).__init__()
        self.conv = nn.Conv2d(3, CHANNELS[step], 1)

    def forward(self, x) :
        return self.conv(x)
         

class Discriminator(nn.Module) :
    def __init__(self, block_count = 9) :
        super(Discriminator, self).__init__()
        self.block = nn.ModuleDict()
        self.from_RGB = nn.ModuleDict()

        for i in range(block_count, 0, -1) :
            self.from_RGB[str(i)] = FromRGB(i)
            self.block[str(i)] = DBlock(i)


    def forward(self, x, step) :
        x = self.from_RGB[str(step)](x)

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
        self.leaky1 = nn.LeakyReLU()

        if self.step != 1 :
            self.conv2 = nn.Conv2d(self.channel, self.next_channel, 3, padding=1) 
            self.leaky2 = nn.LeakyReLU()
            self.avgpool = nn.AvgPool2d(2)

        else :
            self.conv2 = nn.Conv2d(self.channel, self.channel, 4, padding=0)
            self.leaky2 = nn.LeakyReLU()
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
            # minibatch standard deviation
            ################################
            x = x.view(x.shape[0], -1)
            x = self.fc(x)

        return x
            

# Latent space 정의 
class MappingNet(nn.Module) :
    def __init__(self) :
        super(MappingNet, self).__init__()

        self.dense = nn.ModuleList([nn.Linear(Z_SIZE, MAPPING_UINT)])

        for i in range(7) :
            self.dense.append(nn.Linear(MAPPING_UINT, MAPPING_UINT))

    def forward(self, x) :
        # 추후 조절을 위해 입력을 따로 받음
        # 입력은 (100,1) 의 Unifrom-Dist 벡터
        # x = F.normalize(z)
        for i in range(8) :
            x = self.dense[i](x)
            x = nn.ReLU()(x)

        return x


if __name__ == "__main__" :
    z = torch.rand(100)
 
    g = Generator(2)
    d = Discriminator()

    step = 3

    y = g(z, step)
    print(y.shape)

    z = d(y, step)
    print(z.shape)


