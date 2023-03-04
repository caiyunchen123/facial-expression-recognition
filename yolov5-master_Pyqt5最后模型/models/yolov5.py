import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from models.model_part import *
from models.backbone_head import *
from models.detector import *

def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True

class yolov5(nn.Module):
    def __init__(self, ch=3, nc=None, anchors=None):
        super(yolov5, self).__init__()
        self.backbone = backbone_head()
        self.detector = detector()
        s = 256
        self.detector.stride = torch.tensor([s/x.shape[-2] for x in self.forward(torch.zeros(1, ch, s, s))])
        self.detector.anchors /= self.detector.stride.view(-1, 1, 1)

        self.stride = self.detector.stride
        self._initialize_biases()
        initialize_weights(self.backbone)
        initialize_weights(self.detector)


    def forward(self, x):
        y, dt = [], []
        x1 = self.backbone(x)
        x2 = self.detector(x1)
        return x2

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        m = self.detector  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:5 + m.nc] += math.log(0.6 / (m.nc - 0.99999)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

if __name__ == '__main__':
    im = torch.ones(2, 3, 1111, 1111)
    model = yolov5()
    out = model(im)
    print(1)