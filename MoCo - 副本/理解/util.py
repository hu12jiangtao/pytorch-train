import PIL.Image as Image
import torch
import torchvision
import autoaugment
import random
from torchvision import transforms
import numpy as np

def dataset_info(name='image_net'): # 此函数用来获取图片的大小和均值方差
    if name == 'cifar':
        return 32, (0.4914, 0.4822, 0.4465), (0.2471, 0.2435, 0.2616)
    if name == 'image_net':
        return 224, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)

class RandAugment:
    def __init__(self,n=2):
        self.n = n
        self.transform_lst = autoaugment.augment_list()
    def __call__(self,image):
        choice_transform = random.choices(self.transform_lst,k=self.n)
        for type_transform,min_val,max_val in choice_transform:
            m = np.random.uniform(0, 1)
            val = m * float(max_val - min_val) + min_val
            image = type_transform(image, val)
        image = autoaugment.Cutout(image, 0.2) # 将图片中的一块内容进行随机的遮掩
        return image


def get_transform(image_size, mean, std, mode='train', to_tensor=True):
    transform_to_tensor = [torchvision.transforms.ToTensor(),torchvision.transforms.Normalize(mean=mean,std=std)] \
                          if to_tensor is True else []
    if mode == 'train':
        # RandomResizedCrop(image_size)代表的意思为首先随机采集不同的大小和宽高比的图片，然后放大至image_size
        train_transform = [torchvision.transforms.RandomHorizontalFlip(),
                           torchvision.transforms.RandomCrop(image_size, padding=int(image_size * 0.125), padding_mode='reflect'),
                           RandAugment() if image_size < 128 else [torchvision.transforms.RandomResizedCrop(image_size),
                                                                   torchvision.transforms.RandomHorizontalFlip()]]
        return torchvision.transforms.Compose(train_transform + transform_to_tensor)
    elif mode == 'test':
        test_transform = [] if image_size < 128 else [torchvision.transforms.Resize(image_size + 16, interpolation=3),
                                                      torchvision.transforms.CenterCrop(image_size)]
        return torchvision.transforms.Compose(test_transform + transform_to_tensor)

def custom_dataset(base_dataset):
    class Create(base_dataset):
        def __init__(self,*args,**kwargs):
            super(Create, self).__init__(*args,**kwargs)

        def __getitem__(self, item):
            image, label = self.data[item], self.targets[item]
            image = Image.fromarray(image)
            anchor = self.transform(image)
            positive = self.transform(image)
            return anchor, positive, label
    return Create
