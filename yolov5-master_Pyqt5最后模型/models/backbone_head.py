import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_part import *
from models.backbone import *

class backbone_head(nn.Module):
    def __init__(self):
        super(backbone_head, self).__init__()
        inc = 3
        blayers = [3, 6, 9, 3]
        c = 64
        n_c = [c*2**(i+1) for i in range(len(blayers))]
        self.backbone = backbone(inc=inc, blayers=[3, 6, 9, 3])

        self.sppfout = Conv(n_c[-1], n_c[-2], 1, 1, 0)
        self.c3up_1 = C3(n_c[-1], n_c[-2], 3)
        self.c3up_1out = Conv(n_c[-2], n_c[-3], 1, 1, 0)
        self.c3up_2 = C3(n_c[-2], n_c[-3], 3)
        self.c3up_3 = C3(n_c[-3], n_c[-4], 3)

        self.cdown_1 = Conv(n_c[-3], n_c[-3], 3, 2, 1)
        self.c3down_1 = C3(n_c[-2], n_c[-2], 3)
        self.cdown_2 = Conv(n_c[-2], n_c[-2], 3, 2, 1)
        self.c3down_2 = C3(n_c[-1], n_c[-1], 3)


    def forward(self, x):
        out = []
        xb = self.backbone(x)
        x3 = self.sppfout(xb[-1])
        x = self.c3up_1(torch.cat((F.adaptive_avg_pool2d(F.interpolate(x3, scale_factor=2), xb[-3].shape[2:]), xb[-3]), 1))
        x2 = self.c3up_1out(x)
        x1 = self.c3up_2(torch.cat((F.adaptive_avg_pool2d(F.interpolate(x2, scale_factor=2), xb[-4].shape[2:]), xb[-4]), 1))

        x2 = self.c3down_1(torch.cat((F.adaptive_avg_pool2d(self.cdown_1(x1), x2.shape[2:]), x2), 1))
        x3 = self.c3down_2(torch.cat((F.adaptive_avg_pool2d(self.cdown_2(x2), x3.shape[2:]), x3), 1))
        return [x1, x2, x3]

if __name__ == '__main__':
    im = torch.ones(2, 3, 320, 320)
    model = backbone_head()
    out = model(im)
    print(1)
