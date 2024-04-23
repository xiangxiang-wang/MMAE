# data loader CVC14
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import os
import random


# Multi-Focus-Dataset：DUTS
class CVC14(Dataset):
    def __init__(self, dataset_dir, transforms_):
        self.dataset_dir = dataset_dir
        self.dataset_dir1 = self.dataset_dir + '/Visible/FramesPos'
        self.dataset_dir2 = self.dataset_dir + '/FIR/FramesPos'
        self.file_list1 = os.listdir(self.dataset_dir1)
        self.file_list2 = os.listdir(self.dataset_dir2)
        self.transform = transforms.Compose(transforms_)


    def __len__(self):
        return len(self.file_list1)

    def __getitem__(self, item):

        img1 = Image.open(os.path.join(self.dataset_dir1, self.file_list1[item])).convert('RGB')
        img2 = Image.open(os.path.join(self.dataset_dir2, self.file_list1[item])).convert('RGB')

        # img1 = img1.resize((64, 64), Image.BICUBIC)
        # img2 = img2.resize((64, 64), Image.BICUBIC)
        # label = label.resize((64, 64), Image.BICUBIC)

        img1 = self.transform(img1)
        img2 = self.transform(img2)
        return img1, img2


# class DataTest(Dataset):
#     def __init__(self, testData_dir, transforms_):
#         self.testData_dir = testData_dir  # 测试文件路径
#         self.file_list = os.listdir(testData_dir)  # 用于返回指定的文件夹包含的文件或文件夹的名字的列表
#         self.transform = transforms.Compose(transforms_)  # 串联多个transform操作
#
#     def __getitem__(self, idx):
#         image1 = Image.open(self.testData_dir + "/" + self.file_list[idx] + "/" + self.file_list[idx] + "-A.jpg")
#         image2 = Image.open(self.testData_dir + "/" + self.file_list[idx] + "/" + self.file_list[idx] + "-B.jpg")
#         image1 = self.transform(image1)
#         image2 = self.transform(image2)
#         return image1, image2
#
#     def __len__(self):
#         return len(self.file_list)

