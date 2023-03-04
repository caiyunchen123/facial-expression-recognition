import argparse
import os
import numpy as np
import math

import torchvision.models
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.utils import tensorboard
from scipy.stats import wasserstein_distance as wd

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=5, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.00001, help="adam: learning rate")
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
        self.adv_layer = nn.Sequential(nn.Linear(1000, 1),nn.Sigmoid())#nn.sigmoid
        self.aux_layer = nn.Sequential(nn.Linear(1000, opt.n_classes), nn.Softmax(dim=1))
    def forward(self,x):
        out = self.vgg16(x)
        out = out.view(out.shape[0], -1)
        validity = self.adv_layer(out)
        label = self.aux_layer(out)

        return validity,label


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

class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, smoothing=0.2):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, x, target):
        # 计算类别数量
        n_classes = x.size(1)
        # 生成目标张量
        target = target.unsqueeze(1)
        # 生成标签张量
        one_hot = torch.zeros_like(x)
        one_hot.fill_(self.smoothing / (n_classes - 1))
        one_hot.scatter_(index=target,src=self.confidence,dim=1 )
        # 计算交叉熵损失
        log_prb = nn.functional.log_softmax(x, dim=1)
        loss = -(one_hot * log_prb).sum(dim=1).mean()
        return loss
label_loss = LabelSmoothingCrossEntropy()

def loss_func(recon_x, x, mu, logvar):
    # BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    MSE = F.mse_loss(recon_x,x)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # ll = nn.CrossEntropyLoss()(output,labels)
    # HSE = label_loss(recon_x,x)
    return MSE + KLD

class Classifier(nn.Module):
    def __init__(self) -> None:
        super(Classifier, self).__init__()
        self.module = nn.Sequential(
            nn.Linear(64, 256),
            nn.Linear(256, 7),
            nn.Softmax(dim=1)
        )

    def forward(self,x):
        x = self.module(x)
        return x

class CAL(nn.Module):
    def __init__(self):
        super(CAL, self).__init__()
        self.cal = Classifier()
        self.get = ResNet18Enc().ResNet18
    def forward(self,x):
        x = self.get(x)
        x = self.cal(x)
        return x
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
classifier = CAL()
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
    label_loss.cuda()

writer = SummaryWriter('./log')

transform = transforms.Compose([transforms.Resize([224, 224]),transforms.ToTensor(),
                                # transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                                # gray -> GRB 3 channel (lambda function)
                                ])  # for grayscale images
# dataloader
os.makedirs("../../data/fer2013", exist_ok=True)
dataloader = torch.utils.data.DataLoader(
    datasets.ImageFolder(
        "../../data/fer2013/Training",
        transform=transform
        ),
    batch_size=opt.batch_size,
    shuffle=True,
)

optimizer_G = torch.optim.Adam(generator.parameters(), lr=1e-4, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
# optimizer_c = torch.optim.SGD(classifier.parameters(),lr=opt.lr)

FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor

# ----------
#  Training
# ----------
for epoch in range(opt.n_epochs):
    accuracy_sum = 0
    for i,(imgs, labels) in enumerate(dataloader):

            # imgs=torch.reshape(imgs,(256,1,48,48))
            imgs = imgs.cuda()
            labels = labels.cuda()
            batch_size = imgs.shape[0]

            # Adversarial ground truths
            valid = Variable(FloatTensor(batch_size, 1).fill_(0.8), requires_grad=False)
            fake = Variable(FloatTensor(batch_size, 1).fill_(0.2), requires_grad=False)

            # Configure input
            real_imgs = Variable(imgs.type(FloatTensor))
            labels = Variable(labels.type(LongTensor))

            #训练 discriminator
            optimizer_D.zero_grad()

            output,_ = discriminator(imgs)
            # errD_real = adversarial_loss(output,valid)
            errD_real = adversarial_loss(output,valid)
            errD_real.backward()

            real_data_score = output.mean().item()

            z = torch.randn(batch_size,256).cuda()
            # fake_data = z.view(z.shape[0],-1,7,7)
            fake_data = generator.generator_2(z)
            output,_ = discriminator(fake_data)
            # errD_fake = adversarial_loss(output,fake)
            errD_fake = adversarial_loss(output, fake)
            errD_fake.backward()

            fake_data_score = output.data.mean()
            errD = errD_real + errD_fake
            optimizer_D.step()


            recon_x, mu, logvar = generator.forward(real_imgs)
            # feature = generator.generator_1.ResNet18(imgs)
            # output = classifier(imgs)
            optimizer_G.zero_grad()
            loss = loss_func(recon_x, real_imgs, mu,logvar)
            loss.backward(retain_graph=True)
            # accuracy = (output.argmax(1) == labels).sum()
            # accuracy_sum += accuracy
            optimizer_G.step()


            optimizer_G.zero_grad()
            recon_x, mu, logvar = generator.forward(real_imgs)
            valid = Variable(FloatTensor(batch_size, 1).fill_(0.8), requires_grad=False).to(device)
            output,_ = discriminator(recon_x)
            err_vae = adversarial_loss(output,valid)
            err_vae.backward()
            # d_g_z2 = output.mean().item()
            optimizer_G.step()

            # optimizer_c.zero_grad()
            #
            # feature = generator.generator_1.ResNet18(imgs)
            # output = classifier(feature)
            # loss_c = auxiliary_loss(output,labels)
            # loss_c.backward()
            # optimizer_c.step()



            if i%10==0:
                print('[%d/%d][%d/%d] real_score: %.6f fake_score: %.6f '
                      'Loss_D: %.6f Loss_G: %.6f'
                      % (epoch, opt.n_epochs, i, len(dataloader),
                         real_data_score,
                         fake_data_score,
                         errD,err_vae))
                writer.add_scalar("real_score",real_data_score,i)
                writer.add_scalar("fake_score",fake_data_score,i)
                writer.add_scalar("d_loss",errD,i)
                writer.add_scalar("g_loss",err_vae,i)
            # if i%10==0:
            #     print('------------accuracy: %.6f-----------'%(accuracy_sum/40))
            #     accuracy_sum = 0
            if i%100==0:
                sample_z = torch.randn(batch_size, 256).cuda()
            # fake_data = z.view(z.shape[0],-1,7,7)
                fake_data = generator.generator_2(sample_z)
                output,_ = discriminator(fake_data)
                save_image(fake_data, './img_VAE-GAN/fake_images-{}.png'.format(i + 1))
torch.save(generator.state_dict(), './img_VAE-GAN/VAE-GAN-VAE_epoch5_3.pth')
torch.save(discriminator.state_dict(), './img_VAE-GAN/VAE-GAN-Discriminator_epoch5_3.pth')
writer.close()


            # # -----------------
            # #  Train Generator
            # # -----------------
            #
            #
            #
            # # Sample noise and labels as generator input
            # # z = Variable(FloatTensor(np.random.normal(0, 1, (batch_size, opt.latent_dim))))
            #
            # # z = Variable(FloatTensor(np.random.choice(imgs.shape[0],size=(batch_size, opt.latent_dim))))
            # gen_imgs = generator(real_imgs)
            # gen_labels = Variable(LongTensor(np.random.randint(0, opt.n_classes, batch_size)))
            #
            # # Generate a batch of images
            # # gen_imgs = generator(z, gen_labels)
            #
            # # Loss measures generator's ability to fool the discriminator
            # validity, pred_label = discriminator(gen_imgs)
            # g_loss = 0.5 * (adversarial_loss(validity, valid) + auxiliary_loss(pred_label, gen_labels))
            #
            # g_loss.backward(retain_graph = True)
            # optimizer_G.step()
            #
            # # ---------------------
            # #  Train Discriminator
            # # ---------------------
            #
            # optimizer_D.zero_grad()
            #
            # # Loss for real images
            # real_pred, real_aux = discriminator(real_imgs)
            # d_real_loss = (adversarial_loss(real_pred, valid) + auxiliary_loss(real_aux, labels)) / 2
            #
            # # Loss for fake images
            # fake_pred, fake_aux = discriminator(gen_imgs.detach())
            # d_fake_loss = (adversarial_loss(fake_pred, fake) + auxiliary_loss(fake_aux, gen_labels)) / 2
            #
            # # Total discriminator loss
            # d_loss = (d_real_loss + d_fake_loss) / 2
            #
            # # Calculate discriminator accuracy
            # pred = np.concatenate([real_aux.data.cpu().numpy(), fake_aux.data.cpu().numpy()], axis=0)
            # gt = np.concatenate([labels.data.cpu().numpy(), gen_labels.data.cpu().numpy()], axis=0)
            # d_acc = np.mean(np.argmax(pred, axis=1) == gt)
            #
            #
            # d_loss.backward(retain_graph = True)
            # optimizer_D.step()
            #
            # # ---------------------
            # #  Train Classifier
            # # ---------------------
            #
            # optimizer_c.zero_grad()
            #
            # output = classifier(feature_get(real_imgs))
            # loss = auxiliary_loss(output,labels)
            # accuracy = (output.argmax(1) == labels).sum()
            # loss.backward()
            # print(accuracy)
            # optimizer_c.step()
            #
            # writer.add_scalar("g_loss",g_loss,global_step=i)
            # writer.add_scalar("d_loss",d_loss,i)
            # writer.add_images("g_images",gen_imgs,i)
            # writer.add_scalar("accuracy",accuracy,i)
            #
            # print(
            #     "[Epoch %d/%d]  [D loss: %f, acc: %d%%] [G loss: %f]"
            #     % (epoch, opt.n_epochs, d_loss.item(), 100 * d_acc, g_loss.item())
            # )
            # i += 1

# epoch_num = 150
# batch_size = 16
# vae = VAE(z_dim=256).cuda()
# optimizer = optim.Adam(vae.parameters(), lr=1e-4)
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# root = "./dataset/"
#
#
#
# # MNIST dataset (images and labels)
# MNIST_train_dataset = torchvision.datasets.MNIST(root='./data', train=True, transform=transform, download=True)
# MNIST_test_dataset = torchvision.datasets.MNIST(root='./data', train=False, transform=transform)
#
# # Data loader (input pipeline)
# train_iter = torch.utils.data.DataLoader(dataset=MNIST_train_dataset, batch_size=batch_size, shuffle=True)
# test_iter = torch.utils.data.DataLoader(dataset=MNIST_test_dataset, batch_size=batch_size, shuffle=False)
#
# for epoch in range(0, epoch_num):
#     l_sum = 0
#     scheduler.step()
#     for x, y in train_iter:
#         # x = torch.sigmoid(x).cuda()
#         x = x.cuda()
#         print(x.requires_grad)
#         optimizer.zero_grad()
#         recon_x, mu, logvar = vae.forward(x)
#         loss = loss_func(recon_x, x, mu, logvar)
#         l_sum += loss
#         loss.backward()
#         optimizer.step()
#     print("loss\n", l_sum)
#     print(epoch, "\n")
#
# i = 0
# with torch.no_grad():
#     for t_img, y in test_iter:
#         t_img = Variable(t_img).cuda()
#         result, mu, logvar = vae.forward(t_img)
#         utils.save_image(result.data, str(i) + '.png', normalize=True)
#         i += 1






