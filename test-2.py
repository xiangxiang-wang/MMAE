import torch
from data_loader import DataTest
import argparse
from modules2 import MMAE
from torch.utils.data import DataLoader
from torchvision import transforms
import time
import matplotlib.pyplot as plt
import numpy as np

from PIL import Image
import os

from utils import test_save_images


def test(test_dataloader, args):
    device = args.device
    # set_dir = args.testData_dir
    # set_list = os.listdir(set_dir)
    # 获取文件名，去掉后缀
    file_list = os.listdir(args.testData_dir)
    temp_dir = os.listdir(os.path.join(args.testData_dir, file_list[0]))
    set_list = []
    for i in temp_dir:
        portion = os.path.splitext(i)  # 把文件名拆分为名字和后缀

        set_list.append(portion[0])
    print(set_list)
    i = 0
    net = MMAE(args).to(device)
    checkpoint = torch.load(args.saveModel_dir)
    net.load_state_dict(checkpoint['state_dict'], strict=False)
    # pretrained_checkpoint_path = r'E:\MMAE\premask\epoch_140_loss_2.950339.pth'
    # pretrained_net = torch.load(pretrained_checkpoint_path)
    # net.load_state_dict(pretrained_net['state_dict'], strict=False)
    # net.load_state_dict(pretrained_net['net'], strict=False)
    # print("加载预训练模型成功")
    net.eval()
    print(net)
    t1 = time.time()
    for i_test, (image1, image2) in enumerate(test_dataloader):
        image1 = image1.to(device)
        image2 = image2.to(device)
        out, pre_mask = net(image1, image2)
        out1 = out + 2 * (image1 - image2)
        out2 = out + 1.7 * (image2 - image1)
        # out2 = torch.max(out2, image2)
        # out1 = torch.mean(out1, dim=1, keepdim=True)
        # out2 = torch.mean(out2, dim=1, keepdim=True)
        # out2 = 1 * out
        # out = out2
        # out2 = 2 * torch.max(out,image2)
        # out = torch.abs(image2 - image1)
        # out = image1 - image2
        # out = 0.8*image1 + 0.2*image2
        # out = torch.max(image1,image2)
        # out, _ = net(out1, image2)

        EPSILON = 1e-10
        mask1 = torch.exp(out1) / (torch.exp(out2) + torch.exp(out1) + EPSILON)
        mask2 = torch.exp(out2) / (torch.exp(out1) + torch.exp(out2) + EPSILON)
        out1 = mask1 * out1
        # # # # # # # sampled_images1 = mask1 * image1
        # # # # # # # #
        out2 = mask2 * out2
        # out = out2
        # #
        # out2 = 1.5 * image2

        EPSILON = 1e-10
        mask1 = torch.exp(image1) / (torch.exp(image2) + torch.exp(image1) + EPSILON)
        mask2 = torch.exp(image2) / (torch.exp(image1) + torch.exp(image2) + EPSILON)
        # # # # # # # # # # # # print(mask1)
        # # # # # # # # # # # # # print(mask2)
        out1 = mask1 * out1
        # # # # # # sampled_images1 = mask1 * image1
        # # # # # # #
        out2 = mask2 * out2
        # out = out2

        # EPSILON = 1e-10
        # mask1 = torch.exp(out1) / (torch.exp(out2) + torch.exp(out1) + EPSILON)
        # mask2 = torch.exp(out2) / (torch.exp(out1) + torch.exp(out2) + EPSILON)
        # out1 = mask1 * out1
        # # # # # # # sampled_images1 = mask1 * image1
        # # # # # # # #
        # out2 = mask2 * out2
        #
        out = torch.max(out1,out2)
        # out = out1 + out2
        # out = (out1 + out2)/2
        # out =out2
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
    # parser.add_argument('--testData_dir', type=str, default=r"F:\test_dataset\lytro2")
    # parser.add_argument('--testData_dir', type=str, default=r"F:\test_dataset\Roadscene_select2")
    parser.add_argument('--testData_dir', type=str, default=r"F:\test_dataset\Harvard_select2")
    # parser.add_argument('--testData_dir', type=str, default="D:/DeepLearing/set/MFFW")
    parser.add_argument('--saveModel_dir', type=str, default='E:\MMAE\save_model_cvc14\epoch_152_loss_0.166528.pth')
    # parser.add_argument('--saveModel_dir', type=str, default='E:\MMAE\save_model\epoch_82_loss_6.166936.pth')
    parser.add_argument('--result', type=str, default=r'D:\BaiduSyncdisk\MMAE\result/')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    # transforms_ = [transforms.Resize(size=(args.image_size_h, args.image_size_w)), transforms.ToTensor()]
    # transforms_ = [transforms.Resize(size=(64, 64)), transforms.ToTensor()]
    transforms_ = [transforms.ToTensor()]
    test_set = DataTest(testData_dir=args.testData_dir, transforms_=transforms_)
    test_dataloader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=1)
    test(test_dataloader, args)
