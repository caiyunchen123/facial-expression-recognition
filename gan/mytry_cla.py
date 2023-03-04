import argparse
import os
import random

import numpy as np
import math

import torchvision.models
import torchvision.transforms as transforms
from torch.backends import cudnn
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils import tensorboard

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=100, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.001, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--n_classes", type=int, default=7, help="number of classes for dataset")
parser.add_argument("--img_size", type=int, default=48, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=400, help="interval between image sampling")
opt = parser.parse_args()
print(opt)

cuda = True if torch.cuda.is_available() else False
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
#用上采样加卷积代替了反卷积
class ResizeConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, scale_factor, mode='nearest'):
        super().__init__()
        self.scale_factor = scale_factor
        self.mode = mode
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=1, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)
        x = self.conv(x)
        return x


class ResNet18Enc(nn.Module):
    def __init__(self, z_dim=32):
        super(ResNet18Enc, self).__init__()
        self.z_dim = z_dim
        self.ResNet18 = torchvision.models.resnet18(pretrained=True)
        self.num_feature = self.ResNet18.fc.in_features
        self.ResNet18.fc = nn.Linear(self.num_feature, 2 * self.z_dim)

    def forward(self, x):
        x = self.ResNet18(x)
        mu = x[:, :self.z_dim]
        logvar = x[:, self.z_dim:]
        return mu, logvar


class BasicBlockDec(nn.Module):

    def __init__(self, in_planes, stride=1):
        super().__init__()

        planes = int(in_planes / stride)

        self.conv2 = nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_planes)

        if stride == 1:
            self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential()
        else:
            self.conv1 = ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride)
            self.bn1 = nn.BatchNorm2d(planes)
            self.shortcut = nn.Sequential(
                ResizeConv2d(in_planes, planes, kernel_size=3, scale_factor=stride),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        x = self.conv2(x)
        x = self.bn2(x)
        out = torch.relu(x)
        out = self.bn1(self.conv1(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out


class ResNet18Dec(nn.Module):

    def __init__(self, num_Blocks=[2, 2, 2, 2], z_dim=48, nc=3):
        super().__init__()
        self.in_planes = 512

        self.linear = nn.Linear(z_dim, 512)

        self.layer4 = self._make_layer(BasicBlockDec, 256, num_Blocks[3], stride=2)
        self.layer3 = self._make_layer(BasicBlockDec, 128, num_Blocks[2], stride=2)
        self.layer2 = self._make_layer(BasicBlockDec, 64, num_Blocks[1], stride=2)
        self.layer1 = self._make_layer(BasicBlockDec, 64, num_Blocks[0], stride=1)
        self.conv1 = ResizeConv2d(64, nc, kernel_size=3, scale_factor=2)
        self.leakyrelu = nn.LeakyReLU()

    def _make_layer(self, BasicBlockDec, planes, num_Blocks, stride):
        strides = [stride] + [1] * (num_Blocks - 1)
        layers = []
        for stride in reversed(strides):
            layers += [BasicBlockDec(self.in_planes, stride)]
        self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, z):
        x = self.linear(z)
        x = x.view(z.size(0), 512, 1, 1)
        x = F.interpolate(x, scale_factor=7)
        x = self.layer4(x)
        x = self.layer3(x)
        x = self.layer2(x)
        x = self.layer1(x)
        x = F.interpolate(x, size=(112, 112), mode='bilinear')
        x = self.conv1(x)
        # x = torch.sigmoid(x)
        x = self.leakyrelu(x)
        x = x.reshape(x.size(0), 3, 224, 224)
        return x

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.generator_1 = ResNet18Enc(z_dim=256)
        self.generator_2 = ResNet18Dec(z_dim=256)
        # self.linear = nn.Linear()
    def forward(self,x):
        mean, logvar = self.generator_1(x)
        z = self.reparameterize(mean, logvar)
        x = self.generator_2(z)
        return x, mean, logvar

    @staticmethod
    def reparameterize(mean, logvar):
        std = torch.exp(logvar / 2)  # in log-space, squareroot is divide by two
        epsilon = torch.randn_like(std).cuda()
        return epsilon * std + mean


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.vgg16 = torchvision.models.vgg16(pretrained=True)
        # Output layers
        self.adv_layer = nn.Sequential(nn.Linear(1000, 1), nn.Sigmoid())
        self.aux_layer = nn.Sequential(nn.Linear(1000, opt.n_classes), nn.Softmax(dim=1))
    def forward(self,x):
        out = self.vgg16(x)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)

        return validity


# class VAE(nn.Module):
#     def __init__(self, z_dim):
#         super(VAE, self).__init__()
#         self.encoder = ResNet18Enc(z_dim=z_dim)
#         self.decoder = ResNet18Dec(z_dim=z_dim)
#
#     def forward(self, x):
#         mean, logvar = self.encoder(x)
#         z = self.reparameterize(mean, logvar)
#         x = self.decoder(z)
#         return x, mean, logvar
#
#     @staticmethod
#     def reparameterize(mean, logvar):
#         std = torch.exp(logvar / 2)  # in log-space, squareroot is divide by two
#         epsilon = torch.randn_like(std).cuda()
#         return epsilon * std + mean

def loss_func(recon_x, x, mu, logvar):
    # BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    MSE = F.mse_loss(recon_x,x,reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return MSE + KLD

class Classifier(nn.Module):
    def __init__(self) -> None:
        super(Classifier, self).__init__()
        # self.vgg16 = torchvision.models.vgg16(pretrained=False)
        self.module = nn.Sequential(
            nn.Linear(512, 256),
            nn.Dropout(),
            nn.Linear(256,128),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(128, 7),
            # nn.Softmax(dim=1)
        )

    def forward(self,x):
        x = self.module(x)
        return x

class VGGTest(nn.Module):
    def __init__(self, pretrained=True, numClasses=7):
        super(VGGTest, self).__init__()
        # conv1 1/2
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.relu1_1 = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.relu1_2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # conv2 1/4
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.relu2_1 = nn.ReLU(inplace=True)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.relu2_2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # conv3 1/8
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.relu3_1 = nn.ReLU(inplace=True)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu3_2 = nn.ReLU(inplace=True)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.relu3_3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        # conv4 1/16
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.relu4_1 = nn.ReLU(inplace=True)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu4_2 = nn.ReLU(inplace=True)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu4_3 = nn.ReLU(inplace=True)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # conv5 1/32
        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu5_1 = nn.ReLU(inplace=True)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu5_2 = nn.ReLU(inplace=True)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.relu5_3 = nn.ReLU(inplace=True)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        # load pretrained params from torchvision.models.vgg16(pretrained=True)
        # 从原始的 models.vgg16(pretrained=True) 中预设值参数值。
        if pretrained:
            pretrained_model = torchvision.models.vgg16(pretrained=pretrained)  # 从预训练模型加载VGG16网络参数
            pretrained_params = pretrained_model.state_dict()
            keys = list(pretrained_params.keys())
            new_dict = {}
            for index, key in enumerate(self.state_dict().keys()):
                new_dict[key] = pretrained_params[keys[index]]
            self.load_state_dict(new_dict)

        # 但是至于后面的全连接层，根据实际场景，就得自行定义自己的FC层了。
        self.classifier = nn.Sequential(
            nn.Linear(in_features=512 * 1 * 1, out_features=256),
            nn.Dropout(),
            nn.Linear(in_features=256, out_features=256),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(in_features=256, out_features=numClasses),
        )

    def forward(self, x):  # output: 32 * 32 * 3
        x = self.relu1_1(self.conv1_1(x))  # output: 32 * 32 * 64
        x = self.relu1_2(self.conv1_2(x))  # output: 32 * 32 * 64
        x = self.pool1(x)  # output: 16 * 16 * 64

        x = self.relu2_1(self.conv2_1(x))
        x = self.relu2_2(self.conv2_2(x))
        x = self.pool2(x)

        x = self.relu3_1(self.conv3_1(x))
        x = self.relu3_2(self.conv3_2(x))
        x = self.relu3_3(self.conv3_3(x))
        x = self.pool3(x)

        x = self.relu4_1(self.conv4_1(x))
        x = self.relu4_2(self.conv4_2(x))
        x = self.relu4_3(self.conv4_3(x))
        x = self.pool4(x)

        x = self.relu5_1(self.conv5_1(x))
        x = self.relu5_2(self.conv5_2(x))
        x = self.relu5_3(self.conv5_3(x))
        x = self.pool5(x)

        x = x.view(x.size(0), -1)
        output = self.classifier(x)
        return output

# class reconv_resnet18(nn.Module):
#     def __init__(self):
#         super(reconv_resnet18, self).__init__()
#         self.
#
#     def forward(self,x):
#         output =

feature_get = ResNet18Enc()
generator = Generator()
discriminator = Discriminator()
classifier = Classifier()
# Loss functions
adversarial_loss = torch.nn.BCELoss()
auxiliary_loss = torch.nn.CrossEntropyLoss()
hel = torch.nn.MSELoss()
if cuda:
    generator.cuda()
    discriminator.cuda()
    classifier.cuda()
    feature_get.cuda()
    adversarial_loss.cuda()
    auxiliary_loss.cuda()
    hel.cuda()

writer = SummaryWriter('./output_log')

transform = transforms.Compose([transforms.Resize([224, 224]),transforms.ToTensor(),
                                # transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                                # gray -> GRB 3 channel (lambda function)
                                ])  # for grayscale images
# dataloader
os.makedirs("../../data/fer2013", exist_ok=True)
dataset = datasets.ImageFolder("../../data/fer2013/Training",transform=transform)
dataloader = torch.utils.data.DataLoader(
    datasets.ImageFolder(
        "../../data/fer2013/Training",
        transform=transform
        ),
    batch_size=opt.batch_size,
    shuffle=True,
)
test_dataset = datasets.ImageFolder("../../data/fer2013/PublicTest",transform=transform)
test_dataloader = torch.utils.data.DataLoader(test_dataset,batch_size=opt.batch_size,shuffle=True,)

optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_c = torch.optim.SGD(classifier.parameters(),lr=opt.lr,weight_decay=1e-4)

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

generator.load_state_dict(torch.load('./img_VAE-GAN/VAE-GAN-VAE_epoch5_3.pth'))


for epoch in range(opt.n_epochs):
    correct = 0
    total = 0
    classifier.train()
    for i,(imgs, labels) in enumerate(dataloader):

            # imgs=torch.reshape(imgs,(256,1,48,48))

            imgs = imgs.cuda()
            labels = labels.cuda()
            batch_size = imgs.shape[0]
# Train classifier network
            optimizer_c.zero_grad()
            imgs = Variable(imgs.type(FloatTensor))
            labels = Variable(labels.type(LongTensor))
            feature = generator.generator_1.ResNet18(imgs)
            # mean, logvars = generator.generator_1(imgs)
            # feature = generator.reparameterize(mean,logvars)
            output = classifier(feature)
            # pred = np.concatenate([labels.data.cpu().numpy(), output.data.cpu().numpy()], axis=0)
            # gt = np.concatenate([labels.data.cpu().numpy(), gen_labels.data.cpu().numpy()], axis=0)
            loss = auxiliary_loss(output,labels)
            loss.backward()
            optimizer_c.step()
            cor = (output.argmax(1)==labels).sum()
            correct = correct + cor
            total = total + len(labels)
            if i%10==0:
                print('[%d/%d][%d/%d] loss:%.6f accuracy:%.6f'
                      % (epoch, opt.n_epochs, i, len(dataloader),loss,cor/64))
            # sample_z = torch.randn(8, 256).cuda()
            # fake_data = z.view(z.shape[0],-1,7,7)
            # fake_data = generator.generator_2(sample_z)
            # output = discriminator(fake_data)
                writer.add_scalar("LOSS",loss,i+epoch*len(dataloader))
                writer.add_scalar("accuracy_item",cor/64,i+epoch*len(dataloader))
    print("correct:{}".format(correct))
    print("accuracy:%.6f" % (correct / total))
    writer.add_scalar("ACCURACY", correct/total, epoch + 1)
    print("----------第{}个epoch结束-----------".format(epoch + 1))
    # with torch.no_grad():
    #         # torch.no_grad()
    #         classifier.eval()
    #         test_loss_sum = 0
    #         test_accuracy = 0
    #         correct = 0
    #         total = 0
    #         for i, (imgs, labels) in enumerate(test_dataloader):
    #             # imgs=torch.reshape(imgs,(256,1,48,48))
    #             imgs = imgs.cuda()
    #             labels = labels.cuda()
    #             batch_size = imgs.shape[0]
    #             # Train classifier network
    #             # optimizer_c.zero_grad()
    #             imgs = Variable(imgs.type(FloatTensor))
    #             labels = Variable(labels.type(LongTensor))
    #             total += len(labels)
    #             feature = generator.generator_1.ResNet18(imgs)
    #             output = classifier(feature)
    #             a = output.argmax(1)
    #             # pred = output.data.cpu().numpy()
    #             # gt = labels.data.cpu().numpy()
    #             # accuracy = np.mean(np.argmax(pred, axis=1) == gt)
    #             _, predicted = torch.max(output.data, 1)
    #             # total += labels.size(0)
    #             correct += (predicted == labels).sum().item()
    #             accuracy = ((predicted == labels).sum().item()) / (labels.size(0))
    #             print(accuracy)
    #             # ac_list.append(accuracy)
    #             # accuracy = (a == labels).sum()  # argmax() 为取向量中的最大值，0取列中最大值，1取行中最大值,输出的是序列中的最大值的位置，此程序中即0~9
    #             # # test_loss_sum = test_loss_sum + loss.item()
    #             # test_accuracy += accuracy


writer.close()
