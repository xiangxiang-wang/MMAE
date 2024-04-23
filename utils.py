import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# def patchify(imgs, patch_size, in_ch):
#     """
#     imgs: (N, 3, H, W)
#     x: (N, L, patch_size**2 *3)
#     """
#     p = patch_size
#     assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
#
#     h = w = imgs.shape[2] // p
#     x = imgs.reshape(shape=(imgs.shape[0], in_ch, h, p, w, p))
#     x = torch.einsum('nchpwq->nhwpqc', x)
#     x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * in_ch))
#     return

def image_padding(imgs, patch_size):
    p = patch_size
    img_b, img_ch, img_h, img_w = imgs.shape
    m = img_h % p
    n = img_w % p
    if m != 0:
        imgs = F.pad(imgs, (0, 0, 0, p - m), mode="replicate")
    if n != 0:
        imgs = F.pad(imgs, (0, p - n, 0, 0), mode="replicate")  # 左右上下
    return imgs, m, n


def image_unpadding(imgs, patch_size, m, n):
    p = patch_size
    if n != 0:
        l = n - p
        imgs = imgs[:, :, :, :l]
    if m != 0:
        k = m - p
        imgs = imgs[:, :, :k, :]
    return imgs


# def patchify(imgs, patch_size):
#     """
#     imgs: (N, 3, H, W)
#     x: (N, L, patch_size**2 *3)
#     """
#     p = patch_size
#     # assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
#     img_b, img_ch, img_h, img_w = imgs.shape
#     # m = img_h % p
#     # n = img_w % p
#     # if m !=0:
#     #     imgs = F.pad(imgs, (0,0,0,p-m), mode="replicate")
#     # if n !=0:
#     #     imgs = F.pad(imgs, (0,p-m,0,0), mode="replicate")
#     #
#     # _, _, img_h_, img_w_ = imgs.shape
#     h = w = img_h // p
#     x = imgs.reshape(shape=(img_b, img_ch, h, p, w, p))
#     x = torch.einsum('nchpwq->nhwpqc', x)
#     x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * img_ch))
#     return x
#
#
# def unpatchify(x, patch_size, in_ch):
#     """
#     x: (N, L, patch_size**2 *3)
#     imgs: (N, 3, H, W)
#     """
#     p = patch_size
#     h = w = int(x.shape[1] ** .5)
#     assert h * w == x.shape[1]
#
#     x = x.reshape(shape=(x.shape[0], h, w, p, p, in_ch))
#     x = torch.einsum('nhwpqc->nchpwq', x)
#     imgs = x.reshape(shape=(x.shape[0], in_ch, h * p, h * p))
#     return imgs

def patchify(imgs, patch_size):
    """
    imgs: (N, 3, H, W)
    x: (N, L, patch_size**2 *3)
    """
    p = patch_size
    # assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0
    img_b, img_ch, img_h, img_w = imgs.shape
    # m = img_h % p
    # n = img_w % p
    # if m !=0:
    #     imgs = F.pad(imgs, (0,0,0,p-m), mode="replicate")
    # if n !=0:
    #     imgs = F.pad(imgs, (0,p-m,0,0), mode="replicate")
    #
    # _, _, img_h_, img_w_ = imgs.shape
    # h = w = img_h // p
    x = torch.einsum('nchw->nhwc', imgs)
    x = x.reshape(shape=(imgs.shape[0], img_h * img_w, img_ch))
    return x


def unpatchify(x, patch_size, in_ch, img_h, img_w):
    """
    x: (N, L, patch_size**2 *3)
    imgs: (N, 3, H, W)
    """
    p = patch_size
    # h = w = int(x.shape[1] ** .5)
    h = img_h
    w = img_w
    # assert h * w == x.shape[1]
    # x = x.reshape(shape=(x.shape[0], h, w, in_ch))
    # x = x.reshape(shape=(x.shape[0], h, w, p, p, in_ch))
    # x = torch.einsum('nhwc->nchw', x)
    x = torch.einsum('nlc->ncl', x)
    imgs = x.reshape(shape=(x.shape[0], in_ch, h, w))
    return imgs


# def compute_loss(fusion, img_cat, img_1, img_2, put_type='mean', balance=0.01):
#     loss1 = intensity_loss(fusion, img_1, img_2, put_type)
#     loss2 = structure_loss(fusion, img_cat)
#
#     return loss1 + balance * loss2
def compute_loss(fusion, img_cat):
    loss2 = structure_loss(fusion, img_cat)

    return loss2


#
# def compute_loss(fusion, img_1, img_2, put_type='mean'):
#     loss1 = intensity_loss(fusion, img_1, img_2, put_type)
#
#     return loss1

def create_putative(in1, in2, put_type):
    if put_type == 'mean':
        iput = (in1 + in2) / 2
    elif put_type == 'left':
        iput = in1
    elif put_type == 'right':
        iput = in2
    else:
        raise EOFError('No supported type!')

    return iput


def intensity_loss(fusion, img_1, img_2, put_type):
    inp = create_putative(img_1, img_2, put_type)

    # L2 norm
    loss = torch.norm(fusion - inp, 2)

    return loss


def gradient(x):
    H, W = x.shape[2], x.shape[3]

    left = x
    right = F.pad(x, [0, 1, 0, 0])[:, :, :, 1:]  ##扩充 左右上下
    top = x
    bottom = F.pad(x, [0, 0, 0, 1])[:, :, 1:, :]

    dx, dy = right - left, bottom - top

    dx[:, :, :, -1] = 0
    dy[:, :, -1, :] = 0

    return dx, dy


def create_structure(inputs):
    B, C, H, W = inputs.shape[0], inputs.shape[1], inputs.shape[2], inputs.shape[3]

    dx, dy = gradient(inputs)

    structure = torch.zeros(B, 4, H, W)  # Structure tensor = 2 * 2 matrix

    a_00 = dx.pow(2)
    a_01 = a_10 = dx * dy
    a_11 = dy.pow(2)

    structure[:, 0, :, :] = torch.sum(a_00, dim=1)
    structure[:, 1, :, :] = torch.sum(a_01, dim=1)
    structure[:, 2, :, :] = torch.sum(a_10, dim=1)
    structure[:, 3, :, :] = torch.sum(a_11, dim=1)

    return structure


def structure_loss(fusion, img_cat):
    st_fusion = create_structure(fusion)
    st_input = create_structure(img_cat)

    # Frobenius norm
    loss = torch.norm(st_fusion - st_input)

    return loss


# def forward_loss(imgs, imgs2, pred, mask, mask2):
#     """
#     imgs: [N, 3, H, W]
#     pred: [N, L, p*p*3]
#     mask: [N, L], 0 is keep, 1 is remove,
#     """
#
#     target = imgs
#     target2 = imgs2
#
#     # if self.norm_pix_loss:
#     #     mean = target.mean(dim=-1, keepdim=True)
#     #     var = target.var(dim=-1, keepdim=True)
#     #     target = (target - mean) / (var + 1.e-6)**.5
#
#     loss = (pred - target) ** 2
#     loss2 = (pred - target2) ** 2
#     loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
#     loss2 = loss2.mean(dim=-1)
#     # print(loss.shape)
#     # print(mask.shape)
#     loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
#     loss2 = (loss2 * mask2).sum() / mask2.sum()
#     loss_sum = loss + loss2
#     return loss_sum


def forward_loss(imgs, imgs2, pred, mask, mask2):
    """
    imgs: [N, 3, H, W]
    pred: [N, L, p*p*3]
    mask: [N, L], 0 is keep, 1 is remove,
    """

    target = imgs * mask
    target2 = imgs2 * mask2

    # if self.norm_pix_loss:
    #     mean = target.mean(dim=-1, keepdim=True)
    #     var = target.var(dim=-1, keepdim=True)
    #     target = (target - mean) / (var + 1.e-6)**.5

    loss = (pred - target) ** 2
    loss2 = (pred - target2) ** 2
    loss = loss.mean()  # [N, L], mean loss per patch
    loss2 = loss2.mean()
    # print(loss.shape)
    # print(mask.shape)
    # loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
    # loss2 = (loss2 * mask2).sum() / mask2.sum()
    loss_sum = loss + loss2
    return loss_sum


def create_mask(image1, image2):
    # image1:visible image  image2: infrared image
    image1_0 = image1[:, 0, :, :]
    image1_1 = image1[:, 1, :, :]
    image1_2 = image1[:, 2, :, :]
    image1_0 = torch.unsqueeze(image1_0, dim=1)
    image1_1 = torch.unsqueeze(image1_1, dim=1)
    image1_2 = torch.unsqueeze(image1_2, dim=1)

    image1_c1 = torch.cat([image1_0, image1_0, image1_0], dim=1)
    image_temp_c1 = image2 - image1_c1
    image_temp_c1 = image_temp_c1.clamp_(0, 1)

    image1_c2 = torch.cat([image1_1, image1_1, image1_1], dim=1)
    image_temp_c2 = image2 - image1_c2
    image_temp_c2 = image_temp_c2.clamp_(0, 1)

    image1_c3 = torch.cat([image1_2, image1_2, image1_2], dim=1)
    image_temp_c3 = image2 - image1_c3
    image_temp_c3 = image_temp_c3.clamp_(0, 1)

    return image_temp_c1, image_temp_c2, image_temp_c3


def test_save_images(image, path):
    ndarr = image.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save(path)


# def test_save_images2(image, path):
#     ndarr = image.mul_(255).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
#     im = Image.fromarray(ndarr)
#     im.save(path)
def test_save_images2(image, path):
    ndarr = image.mul_(255).add_(0.5).clamp_(0, 255).to('cpu', torch.uint8).numpy()
    im = Image.fromarray(ndarr)
    im.save(path)