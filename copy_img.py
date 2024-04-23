import os
import shutil

testData_dir = r"F:\download_dataset\medicine\GFP"
file_list = os.listdir(testData_dir)
print(file_list)
l = len(file_list)
print(l)
set_list = []
for i in range(l):
    portion = os.path.splitext(file_list[i])  # 把文件名拆分为名字和后缀
    if i % 2 == 0:
        temp = portion[0][0:-2]
        set_list.append(temp)
print(set_list)
l2 = len(set_list)
print(l2)
for j in range(l2):
    origin_path = testData_dir + "/" + set_list[j] + "-g.jpg"
    origin_path2 = testData_dir + "/" + set_list[j] + "-t.jpg"
    new_file_name = r"F:\download_dataset\medicine\GFP-SET\source_1/" + set_list[j] + ".jpg"
    new_file_name2 = r"F:\download_dataset\medicine\GFP-SET\source_2/" + set_list[j] + ".jpg"
    shutil.copyfile(origin_path, new_file_name)
    shutil.copyfile(origin_path2, new_file_name2)

print("复制完成")
