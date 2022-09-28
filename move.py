import os
import shutil

current_path = 'C:/Users/Administrator/Desktop/1111'
print('当前目录：'+current_path)

filename_list = os.listdir(current_path)

for filename in filename_list:
    try:
        name1= filename.split('.')
        name2 = filename.split('-')

        if name1[1] == 'wav':
            try:
                os.mkdir(current_path + '/' + name2[1])
                print('创建文件夹'+name2[1])
            except:
                pass
            try:
                shutil.move(current_path+'\\'+filename,current_path+'\\'+name2[1])
                print(filename+'转移成功！')
            except Exception as e:
                print('移动失败:' + e)
    except:
        pass

print('整理完毕！')



