#  Harvard的训练
import os
from modules2 import MMAE
import torch
import torch.nn as nn
import argparse
from torch.utils.data import DataLoader
import torch.optim as optim
from data_loader_harvard import Harvard
import torchvision.transforms as transforms
from PIL import Image
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid
from torch.nn import init
import numpy as np
import random
import time
from Loss import LpLssimLoss


def weights_init_xavier(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        init.xavier_normal_(m.weight.data)


def training_setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # 在需要生成随机数据的实验中，每次实验都需要生成数据。
    # 设置随机种子是为了确保每次生成固定的随机数，这就使得每次实验结果显示一致了，有利于实验的比较和改进。使得每次运行该 .py 文件时生成的随机数相同。

    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def train(train_loader, args):
    device = args.device
    n_epochs = args.n_epochs
    model_dir = args.saveModel_dir
    batch_size = args.batch_size
    train_num = 2796
    writer1 = SummaryWriter(log_dir="log/loss")

    # ------- 1. define model --------
    # define the net
    net = MMAE(args).to(device)
    net.apply(weights_init_xavier)

    # define the loss
    # criterion = LpLssimLoss().to(args.device)
    # L1 = torch.nn.L1Loss().to(args.device)
    mse = torch.nn.MSELoss().to(args.device)
    # smoothL1 = torch.nn.SmoothL1Loss().to(args.device)
    ssim = LpLssimLoss().to(args.device)
    l1 = torch.nn.L1Loss().to(args.device)

    # ------- 2. define optimizer --------
    optimizer = optim.Adam(net.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.n_steps, gamma=args.gamma)

    # ------- 3. training process --------
    # ite_num = 0
    running_loss = 0.0
    # ite_num4val = 0

    log_dir = model_dir + "epoch_0_loss_0.pth"  ###若需从断点处继续，需要更改此文件为上一次的epoch

    checkpoint_path = r'E:\MMAE\premask\best.pth'
    pretrained_net = torch.load(checkpoint_path)
    # net.load_state_dict(pretrained_net['state_dict'], strict=False)
    net.load_state_dict(pretrained_net['net'], strict=False)
    # model_dict = net.state_dict()
    # state_dict = {k: v for k, v in pretrained_net.items() if k in model_dict.keys()}
    # model_dict.update(state_dict)
    # net.load_state_dict(model_dict)
    print("加载预训练模型成功")
    frozen_list = []
    # pretrained_net_item = pretrained_net['state_dict']
    pretrained_net_item = pretrained_net['net']
    for name, _ in pretrained_net_item.items():
        # name = 'fusion.'+name
        frozen_list.append(name)
    # for p in net.named_parameters():
    #     p_name = p[0][7:]   # 去掉fusion.initial_result.2.bias 的 fusion.
    #     if p_name in frozen_list:  # 只冻结在名字列表内的权重
    #         p[1].requires_grad = False
    for p in net.named_parameters():
        if p[0] in frozen_list:  # 只冻结在名字列表内的权重
            p[1].requires_grad = False
    # 打印梯度是否冻结
    # for p in net.named_parameters():
    #     print(f"{p[0]}'s requires_grad is {p[1].requires_grad}")
    print("冻结预训练模型成功")

    if os.path.exists(log_dir):
        checkpoint = torch.load(log_dir)
        net.load_state_dict(checkpoint['state_dict'], strict=False)
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        print('加载 epoch {} 成功！'.format(start_epoch))
    else:
        start_epoch = 0
        print('无保存模型，将从头开始训练！')

    for epoch in range(start_epoch, n_epochs):
        net.train()
        t1 = time.time()
        for i, (image1, image2, label) in enumerate(train_loader):
            image1 = image1.to(device)
            image2 = image2.to(device)
            label = label.to(device)
            # image1 = image1.to(device)
            # image2 = image2.to(device)
            # label = label.to(device)
            # label = torch.max(image1,image2)
            # out_ = net(image1, image2)
            #
            # out1 = out_ + 2 * (image1 - image2)
            # out2 = out_ + 2 * (image2 - image1)
            # out = net(out1, out2)
            out, _ = net(image1, image2)

            # loss_mse = mse(out,label)
            # loss = loss_mse

            loss_l1 = l1(out, label)
            loss_ssim = ssim(out, label)
            # loss_gra = compute_loss(out, label)
            loss = 0.2 * loss_l1 + 0.8 * loss_ssim

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # # print statistics
            running_loss += loss.item()
            if (i + 1) % 8 == 0:
                print("[epoch: %3d/%3d, batch: %5d/%5d] train loss: %8f " % (
                    epoch + 1, n_epochs, (i + 1) * batch_size, train_num, loss.item()))
        writer1.add_scalar('训练损失', running_loss, epoch)

        if epoch % 2 == 0:
            check_point = {  # 'state_dict': net.state_dict(),
                'state_dict': net.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'epoch': epoch}
            checkpoint_path = model_dir + "epoch_%d_loss_%3f.pth" % (epoch, running_loss)
            torch.save(check_point, checkpoint_path)

        running_loss = 0.0
        scheduler.step()
        t2 = time.time()
        print(t2 - t1)

    print('-------------Congratulations! Training Done!!!-------------')


def parse_args():
    parser = argparse.ArgumentParser()  # 创建一个解析对象
    parser.add_argument("--in_ch", type=int, default=3, help='rgb is 3,gray is 1')  # 向该对象中添加你要关注的命令行参数和选项
    parser.add_argument("--out_ch", type=int, default=64)  # 每一个add_argument方法对应一个你要关注的参数或选项
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--n_epochs', type=int, default=400, help='number of epochs to train')
    parser.add_argument('--n_steps', type=int, default=200, help='number of epochs to update learning rate')
    parser.add_argument('--gamma', type=float, default=0.1, help='')
    parser.add_argument('--dataset_dir', type=str, default=r"F:\train_dataset\Harvard")
    parser.add_argument('--saveModel_dir', type=str, default=r'E:/MMAE/save_model_harvard/')
    return parser.parse_args()  # 后调用parse_args()方法进行解析，解析成功之后即可使用


if __name__ == '__main__':
    training_setup_seed(1)
    args = parse_args()
    transforms_ = [transforms.Resize((64, 64)),
                   # transforms.RandomHorizontalFlip(p=0.6),
                   transforms.ToTensor()]
    train_set = Harvard(dataset_dir=args.dataset_dir, transforms_=transforms_)
    dataset_dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=4)
    train(dataset_dataloader, args)
