import os
import pdb
def move_file(input_path,output_path):
    file_list = os.listdir(input_path)
    for name in file_list:
        if os.path.isdir(input_path+'/'+name):
            os.mkdir(input_path+i)
        elif os.path.isfile(input_path+'/'+name):
            shutil.rename(input_path+i,output_path+i)


if __name__=='__main__':
    input_name = './'
    output_name = '/home/ncepu-lk/PycharmProjects/smartCity'
    try:
        os.mkdir(output_name)
        move_file(input_name,output_name)
    except:
        move_file(input_name,output_name)
