import torch
from torch import nn
from torch.nn import functional as F


 
# Channels Per Layer
# list        0   1   2   3   4   5   6   7   8    9       
CHANNELS = [512,512,512,512,512,256,128, 64, 32,  16]

# Pixels Per Layer
PIXELS =   [  0,  4,  8, 16, 32, 64,128,256,512,1024]

# Noise Ratio Per Layer
NOISE_PROB = [0,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1, 0.1]

EPSILON = 1e-8

# Latent Size
Z_SIZE = 512

# Mapping Network Units
MAPPING_UNITS = 512




# Mapping Network
class MappingNet(nn.Module) :
    def __init__(self) :
        super(MappingNet, self).__init__()

        # FCL List
        self.dense = nn.ModuleList([nn.Linear(Z_SIZE, MAPPING_UNITS)])
        self.act = nn.LeakyReLU(negative_slope=0.2)

        for i in range(7) :
            self.dense.append(nn.Linear(MAPPING_UNITS, MAPPING_UNITS))

    def forward(self, x) :
        
        for i in range(0,4) :
            rc = x
            x = self.dense[ i*2 ](x)
            x = self.act(x)
            x = self.dense[ i*2 +1 ](x)
            x = rc + x
            x = self.act(x)

        return x




# Define Generator
class Generator(nn.Module) :
    def __init__(self, block_count = 9) :
        super(Generator, self).__init__()

        # Sub Module Create 
        # self.mapping = MappingNet()
        self.block = nn.ModuleDict()
        self.to_RGB = nn.ModuleDict()

        # Const Initialize (4 x 4)
        self.const = torch.ones(1, CHANNELS[1], PIXELS[1], PIXELS[1])
                     

        # Layers
        for i in range(1, block_count+1) :
            # Style Block
            self.block[str(i)] = GBlock(i)

            # To RGB Convert Layer
            self.to_RGB[str(i)] = ToRGB(i)


    def forward(self, w, step, noise = None, alpha=1):
        ###########################################
        # w : Embedded Latent Vector 
        #     shape = ( b, MAPPING_UNITS )
        #   
        # step : Progressive Step 
        #
        # noise : Noise List 
        #     shape = layers * ( h * w )
        # 
        # alpha : Smoothing Parameter ( init 0 to 1 )
        #         0 : Upsample Result
        #         1 : StyleBlock Result 
        ###########################################

        # Get Batch Size
        b, _ = w.shape
        
        # Const Vector Start
        x = self.const.expand(b, CHANNELS[1], 4, 4)

        # Main Generator
        for i in range(1, step+1) :

            if (i == step) and (i != 1) :
                ux = F.interpolate(x, scale_factor=2)
                # ux = F.upsample(x, scale_factor= 2, mode='bilinear')

            # if noise is None :
            noise = torch.randn(1, 1, PIXELS[i], PIXELS[i])

            x = self.block[str(i)]( x, w, noise )

        # To RGB with Smoothing
        if step == 1 :
            y = self.to_RGB[str(i)](x)
            
        else : 
            ux = self.to_RGB[str(i-1)](ux)
            x = self.to_RGB[str(i)](x)
            y = ux * (1 - alpha) + (x * alpha)

        return y

        
                    
# Define Style Block 
class GBlock(nn.Module) :
    def __init__(self, step) :
        super(GBlock, self).__init__()

        # Current Step
        self.step = step

        # Pixel, Channel of Current Layer
        self.pixel = PIXELS[self.step]
        self.prev_channel = CHANNELS[self.step - 1]
        self.channel = CHANNELS[self.step]
        # self.noise_prob = NOISE_PROB[self.step]

        # Main Convolution layer
        self.conv0 = nn.Conv2d(self.prev_channel, self.channel, 3, padding = 1)
        self.conv1 = nn.Conv2d(self.channel, self.channel, 3, padding = 1)

        # Layer Shape 
        self.layer_shape = [2, -1, self.channel, 1, 1]
        self.noise_shape = [1, self.channel, self.pixel, self.pixel]

        # Style Mapping ( mu + sigma ) 
        self.style = nn.Linear(MAPPING_UNITS, 2 * self.channel)

        # Noise Factor Per Channel
        self.noise_factor = nn.Parameter( torch.zeros(1, self.channel, 1, 1 ) )

        # Upsample
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')

        # Activation
        self.act = nn.LeakyReLU(0.2)
          


    def forward(self, x, w, noise) :

        # Not Using Upsample Layer in First Block
        if self.step != 1 :
            x = self.upsample(x)
        
        x = self.conv0(x)

        # Add Noise
        x = x + noise * self.noise_factor #* self.noise_prob
        x = self.act(x)

        # Adaptive Instance Normalization
        x = x - torch.mean(x, dim=(2,3), keepdim=True)
        p = torch.rsqrt(torch.mean(x**2, dim=(2,3), keepdim=True) + EPSILON) 
        x = torch.mul(p,x)
        
        style = self.style(w)
        style = style.view(self.layer_shape)
        x = x * style[0] + style[1]

        # Repeat 
        x = self.conv1(x)

        x = x + noise * self.noise_factor #* self.noise_prob
        x = self.act(x)

        x = x - torch.mean(x, dim=(2,3), keepdim=True)
        p = torch.rsqrt(torch.mean(x**2, dim=(2,3), keepdim=True) + EPSILON) 
        x = torch.mul(p,x)

        style = self.style(w)
        style = style.view(self.layer_shape)
        x = x * style[0] + style[1]

        return x

# ToRGB
class ToRGB(nn.Module) :
    def __init__(self, step) :
        super(ToRGB, self).__init__()
        self.conv = nn.Conv2d(CHANNELS[step] ,3, 1)

    def forward(self, x):
        return self.conv(x)

# FromRGB
class FromRGB(nn.Module) :
    def __init__(self, step) :
        super(FromRGB, self).__init__()
        self.conv = nn.Conv2d(3, CHANNELS[step], 1)

    def forward(self, x) :
        return self.conv(x)
         

# Define Discriminator 
class Discriminator(nn.Module) :
    def __init__(self, block_count = 9) :
        super(Discriminator, self).__init__()
        self.block = nn.ModuleDict()
        self.from_RGB = nn.ModuleDict()

        for i in range(block_count, 0, -1) :
            self.block[str(i)] = DBlock(i)
            self.from_RGB[str(i)] = FromRGB(i)
            


    def forward(self, x, step) :
        # From RGB, Using First Block Only
        x = self.from_RGB[str(step)](x)

        # Main Block
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

        self.act = nn.LeakyReLU(0.2)
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

        # MiniBatch Standard Deviation
        if self.step == 1 :
            x = self.stddev(x)

        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.act(x)

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
        b, c, h, w = x.shape

        # Calculate STD
        y = x - x.mean(dim = 0, keepdim = True)
        y = (y**2).mean(0)
        y = torch.sqrt(y + EPSILON)

        # Average To Single Value
        y = y.mean()
        
        # Expand and Concat Channel
        y = y.expand((b, 1, h, w))
        x = torch.cat([x, y], 1)
        return x



# 테스트
if __name__ == "__main__" :
    z = torch.randn(1, 512)
    print(z.shape)
    step = 6

    m = MappingNet()
    g = Generator(step)
    d = Discriminator(step)

    w = m(z)
    print(w.shape)

    y = g(w, step)
    print(y.shape)

    z = d(y, step)
    print(z.shape)



