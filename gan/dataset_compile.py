from keras.utils import to_categorical
from PIL import Image
import os

num_classes = 7   #表情的类别数目
x_train,y_train,x_val,y_val,x_test,y_test = [],[],[],[],[],[]
from tqdm import tqdm
for i in tqdm(range(num_of_instances)):
    usages_name = usages[i]
    emotions_Str_Nmae = emotions_Str[emotions[i]]
    one_hot_label = to_categorical(emotions[i],num_classes) #标签转换为one-hot编码，以满足keras对于数据的要求
    img = list(map(eval,pixels[i].split(' ')))
    np_img = np.asarray(img)
    img = np_img.reshape(48,48)
    if usages[i] == 'Training':
        x_train.append(img)
        y_train.append(one_hot_label)
    elif usages[i] == 'PrivateTest':
        x_val.append(img)
        y_val.append(one_hot_label)
    else:
        x_test.append(img)
        y_test.append(one_hot_label)
    subfolder = os.path.join(r'E:\data\fer2013',usages_name,emotions_Str_Nmae)
    if not os.path.exists(subfolder):
        os.makedirs(subfolder)
    im = Image.fromarray(img).convert('L')
    im.save(os.path.join(subfolder , (str(i)+'.jpg') ))


