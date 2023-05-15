from PIL import Image


def custom_dataset(base_dataset):
    # name: str = base_dataset.__class__.__name__
    # print(name)

    class CustomDataSet(base_dataset):
        def __init__(self, *args, **kwargs):
            super(CustomDataSet, self).__init__(*args, **kwargs)
            # 由于CustomDataSet继承了datasets.cifar.CIFAR10这个类对象因此在创建CustomDataSet的时候应当也要输入root、transform等参数内容

        def __getitem__(self, index): # 这个函数使得CustomDataSet[0]和原本的DataSet[0]相同，代表着一组的样本
            # 此时的__getitem__函数的目标是使每个批量的图片都进行随机的transform的变换(而不是所有的样本的img_q都是经过同一种transform变换而来的)
            # 此时的img是datasets.cifar.CIFAR10这个图片数据集转换为PIL类型的矩阵(通道数放在最后)
            img, target = self.data[index], int(self.targets[index])
            img = Image.fromarray(img) # 将PIL的矩阵转换为PIL的图片
            # 对于self.transform中RandAugment()这个类对象来说，每次输入图片其数据增强方式都会随机进行选择
            ret_img_q = self.transform(img) # 对图片再次进行数据增强
            ret_img_k = self.transform(img) # 对图片再次进行数据增强(这两次增强是不同的)
            if self.target_transform is not None:
                target = self.target_transform(target)

            return ret_img_q, ret_img_k, target

    return CustomDataSet

# import torchvision.models as models
#
# print(type(models.__dict__['resnet18']))
