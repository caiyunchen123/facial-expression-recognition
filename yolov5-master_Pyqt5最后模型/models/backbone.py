import torch
import torch.nn as nn

from models.model_part import *

class backbone(nn.Module):
    def __init__(self, inc, blayers):
        super(backbone, self).__init__()

        self.inputlayer = Conv(inc, 64, 6, 2, 2)

        cin = 64
        self.c3layers = nn.ModuleList()
        for i, blayer in enumerate(blayers):
            self.c3layers.append(nn.Sequential(Conv(cin, cin*2, 3, 2, 1),
                                               C3(cin * 2, cin * 2, blayer),
                                               ))
            cin *= 2

        self.SPPF = SPPF(cin, cin)

    def forward(self, x):
        out = [self.inputlayer(x)]
        for i in range(4):
            out.append(self.c3layers[i](out[-1]))
        out.append(self.SPPF(out[-1]))
        return out

if __name__ == '__main__':
    im = torch.ones(2, 3, 320, 320)
    model = backbone()
    out = model(im)