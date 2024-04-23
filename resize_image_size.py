import os

from PIL import Image

testData_dir = r"F:\test_dataset\MFFW_select2"
fused_dir = r"D:\DeepLearing\ECNN-master\mffw"

file_list = os.listdir(testData_dir)
temp = os.listdir(os.path.join(testData_dir, file_list[0]))
l = len(temp)

# 获取文件名，去掉后缀
# file_list = os.listdir(testData_dir)
temp_dir = os.listdir(os.path.join(testData_dir, file_list[0]))
set_list = []
for i in temp_dir:
    portion = os.path.splitext(i)  # 把文件名拆分为名字和后缀

    set_list.append(portion[0])
print(set_list)

for idx in range(l):
    temp_dir = os.listdir(os.path.join(testData_dir, file_list[0]))
    image1 = Image.open(testData_dir + "/" + file_list[0] + "/" + temp_dir[idx]).convert('RGB')
    # image2 = Image.open(testData_dir + "/" + file_list[1] + "/" + temp_dir[idx]).convert('RGB')
    # fused_image = Image.open(fused_dir + "/" + temp_dir[idx]).convert('RGB')
    fused_image = Image.open(fused_dir + "/" + set_list[idx]+".png").convert('RGB')
    w, h = image1.size
    # fused_image = fused_image.resize((w, h), Image.BICUBIC)
    fused_image = fused_image.resize((w, h), Image.LANCZOS)
    save_dir = fused_dir + '/' + 'result-new'
    if os.path.exists(save_dir) is False:
        os.mkdir(save_dir)
    # fused_image.save(save_dir + "/" + temp_dir[idx])
    fused_image.save(save_dir + "/" + set_list[idx]+".png")
print("融合图像大小调整完毕")
