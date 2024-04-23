import os

import torch
from data_loader import DataTest
import argparse
from PreModel import PreMask
from torch.utils.data import DataLoader
from torchvision import transforms
import time
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image

from utils import test_save_images, image_padding


def test(test_dataloader, args):
    device = args.device
    # 获取文件名，去掉后缀
    file_list = os.listdir(args.testData_dir)
    temp_dir = os.listdir(os.path.join(args.testData_dir, file_list[0]))
    set_list = []
    for i in temp_dir:
        portion = os.path.splitext(i)  # 把文件名拆分为名字和后缀

        set_list.append(portion[0])
    print(set_list)
    i = 0

    net = PreMask().to(device)

    checkpoint = torch.load(args.saveModel_dir)
    net.load_state_dict(checkpoint['state_dict'])
    # net.load_state_dict(checkpoint['net'])
    # net.load_state_dict(torch.load(args.saveModel_dir))
    net.eval()
    print(net)
    t1 = time.time()
    for i_test, (image1, image2) in enumerate(test_dataloader):
        image1 = image1.to(device)
        image2 = image2.to(device)
        out = net(image1, image2)
        # o1, m1, n1 = image_padding(o1, patch_size=net.patch_size)
        # o2, m2, n2 = image_padding(o2, patch_size=net.patch_size)
        #
        # x1 = net.process1(o1)
        # x2 = net.process1(o2)
        # pre_mask = torch.cat([x1, x2], dim=1)
        # pre_mask = net.process2(pre_mask)

        # pre_mask = torch.mean(out, dim=1, keepdim=True)  ##b,1,h,w
        #
        # pre_mask = net.pool(pre_mask)  ##b,1,h//patch_size,w//patch_size
        # s1 = torch.ones_like(pre_mask)
        # s2 = torch.zeros_like(pre_mask)
        # pre_mask = torch.where(pre_mask > 0.5, s1, s2)  ##b,1,h//patch_size,w//patch_size
        # pre_mask = net.up_sample(pre_mask)  ##b,1,h,w

        # out = torch.cat([pre_mask,pre_mask,pre_mask],dim=1)

        # out_image = transforms.ToPILImage()(torch.squeeze(out.data.cpu(), 0))

        # out_image = out_image.resize((520, 520))  ####重设图像大小

        out = torch.squeeze(out)
        test_save_images(out, args.result + set_list[i] + ".png")
        print(args.result + set_list[i] + ".png")
        i = i + 1
    t2 = time.time()
    print(t2 - t1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_ch", type=int, default=3)
    parser.add_argument("--out_ch", type=int, default=64)
    parser.add_argument("--n_resblocks", type=int, default=3)
    parser.add_argument("--n_convs", type=int, default=3)
    parser.add_argument('--device', type=str, default='cuda:0')
    # parser.add_argument('--testData_dir', type=str, default="DataSet/testData/lytro")
    parser.add_argument('--testData_dir', type=str, default=r"F:\test_dataset\lytro2")
    # parser.add_argument('--testData_dir', type=str, default="D:/DeepLearing/set/MFFW")
    parser.add_argument('--saveModel_dir', type=str, default=r'E:\MMAE\premask\epoch_140_loss_2.950339.pth')
    parser.add_argument('--result', type=str, default=r'D:\BaiduSyncdisk\MMAE\result\test_result/')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    # transforms_ = [transforms.Resize(size=(256, 256)), transforms.ToTensor()]
    transforms_ = [transforms.ToTensor()]
    test_set = DataTest(testData_dir=args.testData_dir, transforms_=transforms_)
    test_dataloader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1)
    test(test_dataloader, args)
