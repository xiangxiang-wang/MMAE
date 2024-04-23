# data loader
from __future__ import print_function, division  # division精确除法
# 如果某个版本中出现了某个新的功能特性，而且这个特性和当前版本中使用的不兼容，也就是它在该版本中不是语言标准，那么我如果想要使用的话就需要从future模块导入。
from PIL import Image
# Python图片处理模块PIL
import torchvision.transforms as transforms  # torchvision.transforms主要是用于常见的一些图形变换，例如裁剪、旋转等。
from torch.utils.data import Dataset, DataLoader
import os
import random


# ==========================dataset load==========================

class SalObjDataset(Dataset):  # 继承Dataset
    def __init__(self, dataset_dir, transforms_, rgb=True):
        self.dataset_dir = dataset_dir  # 路径
        self.file_list = os.listdir(self.dataset_dir)  # 用于返回指定的文件夹包含的文件或文件夹的名字的列表。
        self.transform = transforms.Compose(transforms_)  # 串联多个transform操作
        self.rgb = rgb  # 布尔型变量
        self.mask_path = r"D:/DeepLearing/pythonProject/code/DataSet/ori_data/DUTS-TR/Mask-img"

    def __len__(self):
        return len(self.file_list)  # 返回文件数量

    def __getitem__(self, idx):  # 如果需要类的实例能像list那样按照下标取出元素，需要实现__getitem__()方法
        # 调用对象的属性可以像字典取值一样使用中括号['key']

        # sal data loading
        temp_dir = os.listdir(os.path.join(self.dataset_dir, self.file_list[idx]))  # os.path.join函数用于路径拼接文件路径
        temp_idx = random.randint(0, 3)  # 返回[0,3]之间任意整数
        chird_dir = temp_dir[temp_idx]
        label = Image.open(self.mask_path + "/" + self.file_list[idx] + ".png").convert('RGB')
        if self.rgb == True:
            img1 = Image.open(
                self.dataset_dir + '/' + self.file_list[idx] + "/" + chird_dir + "/" + self.file_list[idx] + "_1.jpg")
            img2 = Image.open(
                self.dataset_dir + '/' + self.file_list[idx] + "/" + chird_dir + "/" + self.file_list[idx] + "_2.jpg")
            # label = Image.open(self.mask_path + "/" + self.file_list[idx] + ".png")
        else:
            img1 = Image.open(self.dataset_dir + '/' + self.file_list[idx] + "/" + chird_dir + "/" + self.file_list[
                idx] + "_1.jpg").convert('L')
            img2 = Image.open(self.dataset_dir + '/' + self.file_list[idx] + "/" + chird_dir + "/" + self.file_list[
                idx] + "_2.jpg").convert('L')
            # label = Image.open(self.mask_path + "/" + self.file_list[idx] + ".png").convert('RGB')  # image.open().convert('L')转换为灰度图像
        # 裁剪为256*256
        # img1 = img1.resize((256, 256), Image.BICUBIC)  # Image.BICUBIC ：三次样条插值
        # img2 = img2.resize((256, 256), Image.BICUBIC)
        # label = label.resize((256, 256), Image.BICUBIC)
        # # 裁剪为64*64
        # img1 = img1.resize((64, 64), Image.BICUBIC)  # Image.BICUBIC ：三次样条插值
        # img2 = img2.resize((64, 64), Image.BICUBIC)
        # label = label.resize((64, 64), Image.BICUBIC)
        # 水平翻转
        if random.random() < 0.5:  # 用于生成一个0到1的随机符点数: 0 <= n < 1.0
            img1 = img1.transpose(Image.FLIP_LEFT_RIGHT)  # 图像左右翻转
            img2 = img2.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)
        img1 = self.transform(img1)  # 调用transform函数
        img2 = self.transform(img2)
        label = self.transform(label)
        return img1, img2, label


class DataTest(Dataset):
    def __init__(self, testData_dir, transforms_):
        self.testData_dir = testData_dir  # 测试文件路径
        self.file_list = os.listdir(testData_dir)  # 用于返回指定的文件夹包含的文件或文件夹的名字的列表
        self.transform = transforms.Compose(transforms_)  # 串联多个transform操作

    def __getitem__(self, idx):
        image1 = Image.open(self.testData_dir + "/" + self.file_list[idx] + "/" + self.file_list[idx] + "-A.jpg")
        image2 = Image.open(self.testData_dir + "/" + self.file_list[idx] + "/" + self.file_list[idx] + "-B.jpg")
        image1 = self.transform(image1)
        image2 = self.transform(image2)
        return image1, image2

    def __len__(self):
        return len(self.file_list)
