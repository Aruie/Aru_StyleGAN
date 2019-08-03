from torch import nn
from torch.nn import functional as F

class Generator(nn.Module) :
    def __init__(self) :
        None
    def forward(self, x, step):
        None


class Discriminator(nn.Module) :
    def __init__(self) :
        None
    def forward(self, x, step) :
        None


class Loss(nn.Module) :
    def __init__(self) :
        None
    def forward(self, x, y) :
        None


class MappingNet(nn.Module) :
    def __init__(self) :
        super(MappingNet, self).__init__()

        self.dense = [(nn.Linear(100, 8))]

        for i in range(7) :
            dense.append = nn.Linear(8, 8)

    def forward(self, z) :
        # 입력값은 (100,1) 의 Unifrom-Dist 벡터
        x = F.normalize(100)(z)
        
        for i in range(8) :
            x = self.dense[i](x)
            x = nn.relu(x)

        return x


if __name__ == 'main' :
    a = Mappingnet()

