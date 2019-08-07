import os
import del_file
b_path = ['/home/ncepu-lk/smartcityData/source_pano/scene1_test/scene1_jiading_lib_test',
	 '/home/ncepu-lk/smartcityData/source_pano/scene2_test/scene2_siping_lib_test',
	 '/home/ncepu-lk/smartcityData/source_pano/scene3_test/scene3_siping_zhonghe_test',
	 '/home/ncepu-lk/smartcityData/source_pano/scene4_test/scene4_jiading_zhixin_test',
	 '/home/ncepu-lk/smartcityData/source_pano/scene5_test/scene5_jiading_rainbow_test',
	 '/home/ncepu-lk/smartcityData/source_pano/scene6_test/scene6_jiading_bolou_test',
	 '/home/ncepu-lk/smartcityData/source_pano/scene7_test/scene7_jiading_riverside_test',
	 '/home/ncepu-lk/smartcityData/source_pano/scene8_test/scene8_jiading_hualou_test']
for index,j in enumerate(b_path):
	bpath = j
	l = os.listdir(bpath)
	npath = '/home/ncepu-lk/smartcityData/pano_test/scene'+str(index+1)
	for i in l:
	  fpath = os.path.join(bpath,i,'thumbnail.jpg')
	  opath = os.path.join(npath,i+'.jpg')
	  os.system('cp '+fpath+' '+opath)
	

