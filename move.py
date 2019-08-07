import os
import del_file
b_path = ['/home/ncepu-lk/smartcityData/source_pano/scene1/scene1_jiading_lib_training',
	 '/home/ncepu-lk/smartcityData/source_pano/scene2/scene2_siping_lib_training',
	 '/home/ncepu-lk/smartcityData/source_pano/scene3/scene3_siping_zhonghe_training',
	 '/home/ncepu-lk/smartcityData/source_pano/scene4/scene4_jiading_zhixin_training',
	 '/home/ncepu-lk/smartcityData/source_pano/scene5/scene5_jiading_rainbow_training',
	 '/home/ncepu-lk/smartcityData/source_pano/scene6/scene6_jiading_bolou_training',
	 '/home/ncepu-lk/smartcityData/source_pano/scene7/scene7_jiading_riverside_training',
	 '/home/ncepu-lk/smartcityData/source_pano/scene8/scene8_jiading_hualou_training']
for index,j in enumerate(b_path):
	bpath = j
	l = os.listdir(bpath)
	npath = '/home/ncepu-lk/smartcityData/pano_train/scene'+str(index+1)
	for i in l:
	  if ('csv' in i):
	     continue
	  fpath = os.path.join(bpath,i,'thumbnail.jpg')
	  opath = os.path.join(npath,i+'.jpg')
	  os.system('cp '+fpath+' '+opath)
	

