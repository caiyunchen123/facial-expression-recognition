from matplotlib import pyplot as plt
from torch import nn
import torch
import torch.optim as optim
from torchvision import datasets, transforms, models
import torchvision
from FER2013 import FER2013
from CK import CK
from swin_model import SwinTransformer
"""
第一步：加载数据集（这里是之前跑的代码，用的数据集是CIFAR10,所以可能要麻烦学长修改一下这里）
"""
batch_size = 64
cut_size = 44

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    # transforms.RandomCrop(cut_size),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.TenCrop(cut_size),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
])

trainset = FER2013(split='Training', transform=transform_train)
# trainset = CK(split='Training', transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=0)
testset = FER2013(split='PublicTest', transform=transform_test)
# testset = CK(split='Testing', transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

print(len(trainset))


"""
第二步：定义VGG16 模型
"""


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


def vgg_train():
    epochs = 200  # 训练次数
    learning_rate = 1e-4  # 学习率

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # net = VGGTest()
    net = SwinTransformer(out_channel=7, patch_size=2, window_size=4, img_size=(48, 48), n_swin_blocks=(2, 2, 6), n_attn_heads=(3, 12, 24))
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失
    optimizer = optim.Adam(net.parameters(), lr=learning_rate, weight_decay=1e-4)  # Adam优化器
    loss_list = []
    net = net.to(device)

    iter = 0
    running_loss = 0.0
    best_loss = 100
    for epoch in range(epochs):  # 迭代
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            # labels: [batch_size, 1]
            # print(i)
            # print(labels)
            # 初始化梯度
            optimizer.zero_grad()

            outputs = net(inputs)  # outputs: [batch_size, 10]
            # print(outputs.shape)
            loss = criterion(outputs, labels)
            loss_list.append(loss.item())
            loss.backward()
            optimizer.step()

            iter += 1
            # 打印loss
            if loss.item() < best_loss:
                torch.save(net.state_dict(), 'model.pth')
                best_loss = loss.item()
            running_loss += loss.item()
            if iter % 100 == 99:  # print loss every 200 mini batch
                print('step/epoch[%d, %8d] loss: %.8f' %
                      (epoch + 1, iter + 1, running_loss / 2000))
                running_loss = 0.0
    step_list = []
    for j in range(len(loss_list)):
        step_list.append(j)
    plt.rcParams['figure.figsize'] = (16.0, 8.0)  # 单位是inches
    plt.plot(step_list, loss_list)
    plt.ylabel('Loss')
    plt.xlabel('')
    plt.show()
    print('Finished Training')

    # torch.save(net, 'model.pt')


    # ac_list = []
    # with torch.no_grad():
    #     correct = 0
    #     total = 0
    #     for images, labels in testloader:
    #         images = images.to(device)
    #         labels = labels.to(device)
    #         outputs = net(images)
    #         _, predicted = torch.max(outputs.data, 1)
    #         total += labels.size(0)
    #         correct += (predicted == labels).sum().item()
    #         accuracy = ((predicted == labels).sum().item()) / (labels.size(0))
    #         ac_list.append(accuracy)
    #     step_list = []
    #     for j in range(len(ac_list)):
    #         step_list.append(j)
    #     plt.rcParams['figure.figsize'] = (16.0, 8.0)  # 单位是inches
    #     plt.plot(step_list, ac_list)
    #     plt.ylabel('accuracy')
    #     plt.xlabel('')
    #     plt.show()
    #     print('The Accuracy of the model on the test images: {} %'.format(100 * correct / total))


if __name__ == '__main__':
    vgg_train()
