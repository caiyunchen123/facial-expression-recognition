from PIL import Image
import numpy as np
import h5py
import torch.utils.data as data


class FER2013(data.Dataset):
    '''
    Args:
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from test set.
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version
    '''

    def __init__(self, split='Training', transform=None):
        self.transform = transform
        self.split = split  # training set or test set
        self.data_source = h5py.File('H5File/FER2013.h5', 'r', driver='core')
        # now load the  picked numpy arrays
        if self.split == 'Training':
            self.train_data = self.data_source['Training_pixel']
            self.targets = self.data_source['Training_label']
            self.train_data = np.asarray(self.train_data)
            self.train_data = self.train_data.reshape((28709, 48, 48))
            self.data = self.train_data

        elif self.split == 'PublicTest':
            self.PublicTest_data = self.data_source['PublicTest_pixel']
            self.targets = self.data_source['PublicTest_label']
            self.PublicTest_data = np.asarray(self.PublicTest_data)
            self.data = self.PublicTest_data.reshape((3589, 48, 48))

        else:
            self.PrivateTest_data = self.data_source['PrivateTest_pixel']
            self.PrivateTest_labels = self.data_source['PrivateTest_label']
            self.PrivateTest_data = np.asarray(self.PrivateTest_data)
            self.data = self.PrivateTest_data.reshape((3589, 48, 48))

    def __getitem__(self, index):
        '''
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        '''
        if self.split == 'Training':
            img, target = self.data[index], self.targets[index]
        elif self.split == 'PublicTest':
            img, target = self.data[index], self.targets[index]
        else:
            img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = img[:, :, np.newaxis]
        img = np.concatenate((img, img, img), axis=2)
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

    def __len__(self):
        if self.split == 'Training':
            return len(self.data)
        elif self.split == 'PublicTest':
            return len(self.data)
        else:
            return len(self.data)


if __name__ == '__main__':
    # import transforms as transforms
    # transform_train = transforms.Compose([
    # 	transforms.ToTensor(),
    # ])
    transform_train = None
    data = FER2013(split='Training', transform=transform_train)
    for i in range(3):
        print(data.__getitem__(i))
    print(data.__len__())
