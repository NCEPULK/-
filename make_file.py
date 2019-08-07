import os

def make_file(path):
	for i in [1,2,3,4,5,6,7,8]:
		os.mkdir(path+'/scene'+str(i))

if __name__=='__main__':
	make_file('/home/ncepu-lk/smartcityData/results')
