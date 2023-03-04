import torchvision.transforms as tf
from main import *

def val():
    # transform_test = transforms.Compose([
    #     transforms.TenCrop(cut_size),
    #     transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
    # ])
    transform_test = tf.Compose([
        tf.ToTensor()
    ])

    batch_size=1

    # testset = FER2013(split='PublicTest', transform=transform_test)
    testset = CK(split='Testing', transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=0)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # model = VGGTest()
    # model = SwinTransformer(out_channel=7, window_size=4, img_size=(48, 48), n_swin_blocks=(2, 6), n_attn_heads=(3, 12))
    model = SwinTransformer(out_channel=7, patch_size=2, window_size=4, img_size=(48, 48), n_swin_blocks=(2, 2, 6), n_attn_heads=(3, 12, 24))
    model.load_state_dict(torch.load('model.pth'))
    model = model.to(device)

    ac_list = []
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in testloader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).cpu().sum().item()
            accuracy = ((predicted == labels).cpu().sum().item()) / (labels.size(0))
            ac_list.append(accuracy)
        step_list = []
        for j in range(len(ac_list)):
            step_list.append(j)
        plt.rcParams['figure.figsize'] = (16.0, 8.0)  # 单位是inches
        plt.plot(step_list, ac_list)
        plt.ylabel('accuracy')
        plt.xlabel('')
        plt.savefig('1.png')
        plt.show()
        print('The Accuracy of the model on the test images: {} %'.format(100 * correct / total))


if __name__ == '__main__':
    val()