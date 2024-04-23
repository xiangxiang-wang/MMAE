import cv2

import rgb_ycbcr_convert

source_img1_path = r"F:\test_dataset\Roadscene_select2\source_1/002.png"
source_img2_path = r"F:\test_dataset\Roadscene_select2\source_2/002.png"
savepath = r"D:\BaiduSyncdisk\Referpaper_second\MMIF-DDFM-main\roadscene-1\recon\result-new\rgb/002.png"

fuse_path = r"D:\BaiduSyncdisk\Referpaper_second\MMIF-DDFM-main\roadscene-1\recon\result-new/002.png"

source_img1 = cv2.imread(source_img1_path, cv2.COLOR_BGR2RGB)
ycrcb_img1 = cv2.cvtColor(source_img1, cv2.COLOR_RGB2YCrCb)  # opencv默认只能变为ycrcb
ycbcr_img1 = ycrcb_img1[:, :, (0, 2, 1)]  # ycrcb变为ycbcr
cb1 = ycbcr_img1[:, :, 1]  # 单独取出cb亮度分量
cr1 = ycbcr_img1[:, :, 2]
#
source_img2 = cv2.imread(source_img2_path, 0)
source_img2 = cv2.cvtColor(source_img2, cv2.COLOR_GRAY2RGB)
ycrcb_img2 = cv2.cvtColor(source_img2, cv2.COLOR_RGB2YCrCb)  # opencv默认只能变为ycrcb
ycbcr_img2 = ycrcb_img2[:, :, (0, 2, 1)]  # ycrcb变为ycbcr
cb2 = ycbcr_img2[:, :, 1]  # 单独取出cb亮度分量
cr2 = ycbcr_img2[:, :, 2]
# cb2 =cb1
# cr2 =cr1

fuseimage = cv2.imread(fuse_path, cv2.COLOR_BGR2RGB)
ycrcb_fuseimage = cv2.cvtColor(fuseimage, cv2.COLOR_RGB2YCrCb)  # opencv默认只能变为ycrcb
print(ycrcb_fuseimage.shape)
fusedcb = rgb_ycbcr_convert.fusecbcr(cb1,cb2)
fusedcr = rgb_ycbcr_convert.fusecbcr(cr1,cr2)
# output = rgb_ycbcr_convert.ycbcr2rgb(output[0,:,:,0].cpu().numpy(),fusedcb,fusedcr)
output = rgb_ycbcr_convert.ycbcr2rgb(ycrcb_fuseimage[:,:,0],fusedcb,fusedcr)
print(output.shape)
cv2.imwrite(savepath, output)
