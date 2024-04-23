# 预训练数据集载入
# data loader
from __future__ import print_function, division  # division精确除法
# 如果某个版本中出现了某个新的功能特性，而且这个特性和当前版本中使用的不兼容，也就是它在该版本中不是语言标准，那么我如果想要使用的话就需要从future模块导入。
from PIL import Image
# Python图片处理模块PIL
import torchvision.transforms as transforms  # torchvision.transforms主要是用于常见的一些图形变换，例如裁剪、旋转等。
from torch.utils.data import Dataset, DataLoader
import os
import random


# class Harvard(Dataset):
#     def __init__(self, dataset_dir, transforms_):
#         self.dataset_dir = dataset_dir
#         self.file_list = os.listdir(self.dataset_dir)
#         self.transform = transforms.Compose(transforms_)
#
#     def __len__(self):
#         return 2 * len(self.file_list)
#
#     def __getitem__(self, idx):
#         # image1 = Image.open(self.testData_dir + "/" + self.file_list[idx] + "/" + self.file_list[idx] + "-A.jpg").convert('RGB')
#         # image2 = Image.open(self.testData_dir + "/" + self.file_list[idx] + "/" + self.file_list[idx] + "-B.jpg").convert('RGB')
#         num = len(self.file_list)
#         if idx < num:
#             image1 = Image.open(
#                 self.dataset_dir + "/" + self.file_list[idx] + "/" + self.file_list[idx] + "_FDG.jpg").convert('RGB')
#             image2 = Image.open(
#                 self.dataset_dir + "/" + self.file_list[idx] + "/" + self.file_list[idx] + "_MR-T1.jpg").convert('RGB')
#             label = Image.open(
#                 self.dataset_dir + "/" + self.file_list[idx] + "/" + self.file_list[idx] + "_T1-FDG.jpg").convert(
#                 'RGB')
#         else:
#             item = idx - num
#             image1 = Image.open(
#                 self.dataset_dir + "/" + self.file_list[item] + "/" + self.file_list[item] + "_FDG.jpg").convert('RGB')
#             image2 = Image.open(
#                 self.dataset_dir + "/" + self.file_list[item] + "/" + self.file_list[item] + "_MR-T2.jpg").convert('RGB')
#             label = Image.open(
#                 self.dataset_dir + "/" + self.file_list[item] + "/" + self.file_list[item] + "_T2-FDG.jpg").convert(
#                 'RGB')
#         image1 = self.transform(image1)
#         image2 = self.transform(image2)
#         label = self.transform(label)
#         return image1, image2, label

class Harvard(Dataset):
    def __init__(self, dataset_dir, transforms_):
        self.dataset_dir = dataset_dir
        self.file_list = os.listdir(self.dataset_dir)
        self.transform = transforms.Compose(transforms_)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        # image1 = Image.open(self.testData_dir + "/" + self.file_list[idx] + "/" + self.file_list[idx] + "-A.jpg").convert('RGB')
        # image2 = Image.open(self.testData_dir + "/" + self.file_list[idx] + "/" + self.file_list[idx] + "-B.jpg").convert('RGB')


        image1 = Image.open(
            self.dataset_dir + "/" + self.file_list[idx] + "/" + self.file_list[idx] + "_FDG.jpg").convert('RGB')
        image2 = Image.open(
            self.dataset_dir + "/" + self.file_list[idx] + "/" + self.file_list[idx] + "_MR-T1.jpg").convert('RGB')
        label = Image.open(
            self.dataset_dir + "/" + self.file_list[idx] + "/" + self.file_list[idx] + "_T1-FDG.jpg").convert(
                'RGB')
        image1 = self.transform(image1)
        image2 = self.transform(image2)
        label = self.transform(label)
        return image1, image2, label

class DataTest(Dataset):
    def __init__(self, testData_dir, transforms_):
        self.testData_dir = testData_dir  # 测试文件路径
        self.file_list = os.listdir(testData_dir)  # 用于返回指定的文件夹包含的文件或文件夹的名字的列表
        self.transform = transforms.Compose(transforms_)  # 串联多个transform操作

    def __getitem__(self, idx):
        # image1 = Image.open(self.testData_dir + "/" + self.file_list[idx] + "/" + self.file_list[idx] + "-A.jpg").convert('RGB')
        # image2 = Image.open(self.testData_dir + "/" + self.file_list[idx] + "/" + self.file_list[idx] + "-B.jpg").convert('RGB')
        image1 = Image.open(
            self.testData_dir + "/" + self.file_list[idx] + "/" + self.file_list[idx] + "_FDG.jpg").convert('RGB')
        image2 = Image.open(
            self.testData_dir + "/" + self.file_list[idx] + "/" + self.file_list[idx] + "_MR-T1.jpg").convert('RGB')
        # image2 = Image.open(
        #     self.testData_dir + "/" + self.file_list[idx] + "/" + self.file_list[idx] + "_MR-T2.jpg").convert('RGB')
        image1 = self.transform(image1)
        image2 = self.transform(image2)
        return image1, image2

    def __len__(self):
        return len(self.file_list)
