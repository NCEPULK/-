import json
import csv
import numpy
import numpy as np
import math
import pdb


def get_martix_M(v0, v1):
    v0 = numpy.array(v0, dtype=numpy.float64, copy=True)
    v1 = numpy.array(v1, dtype=numpy.float64, copy=True)

    ndims = v0.shape[0]
    if ndims < 2 or v0.shape[1] < ndims or v0.shape != v1.shape:
        raise ValueError("input arrays are of wrong shape or type")

    # move centroids to origin
    t0 = -numpy.mean(v0, axis=1)
    M0 = numpy.identity(ndims+1)
    M0[:ndims, ndims] = t0
    v0 += t0.reshape(ndims, 1)
    t1 = -numpy.mean(v1, axis=1)
    M1 = numpy.identity(ndims+1)
    M1[:ndims, ndims] = t1
    v1 += t1.reshape(ndims, 1)
    #pdb.set_trace()

    # Rigid transformation via SVD of covariance matrix
    u, s, vh = numpy.linalg.svd(numpy.dot(v1, v0.T))
    # rotation matrix from SVD orthonormal bases
    R = numpy.dot(u, vh)
    if numpy.linalg.det(R) < 0.0:
        # R does not constitute right handed system
        R -= numpy.outer(u[:, ndims-1], vh[ndims-1, :]*2.0)
        s[-1] *= -1.0
    # homogeneous transformation matrix
    M = numpy.identity(ndims+1)
    M[:ndims, :ndims] = R



    # Affine transformation; scale is ratio of RMS deviations from centroid
    v0 *= v0
    v1 *= v1
    M[:ndims, :ndims] *= math.sqrt(numpy.sum(v1) / numpy.sum(v0))

    # move centroids back
    M = numpy.dot(numpy.linalg.inv(M1), numpy.dot(M, M0))
    M /= M[ndims, ndims]
    return M



def get_transform(X, Xp):
    """Align with GPS and GCP data using direct 3D-3D matches."""
    # Compute similarity Xp = s A X + b
    v0 = numpy.array(X.T, dtype=numpy.float64, copy=False)[:3]
    v1 = numpy.array(Xp.T, dtype=numpy.float64, copy=False)[:3]
    T = get_martix_M(v0, v1)
    A, b = T[:3, :3], T[:3, 3]
    s = np.linalg.det(A)**(1. / 3)
    A /= s
    return s, A, b

def read_file(sfm_data_path,label_path):#get label & predict data

    ######################get_predict_data#############################
    with open(sfm_data_path,"r") as f:
        data = {}  # data{img_name: value}
        all_data = json.loads(f.read()) # all reconstruction imgs

        view_key2value = {}
        extr_key2value = {}
        for img in all_data['views']:
            view_key2value[img['key']] = img['value']

        for img in all_data['extrinsics']:
            extr_key2value[img['key']] = img['value']

        for key in view_key2value:
            try:
                img_name = view_key2value[key]['ptr_wrapper']['data']['filename']
                data[img_name] = extr_key2value[key]
            except:
                pass
    ####################get_label_data################################
    try:
        csv_file = csv.reader(open(label_path,'r',encoding='utf-8'))
        csv_data=[]
        for stu in csv_file:
            csv_data.append(stu)
    except:
        csv_file = csv.reader(open(label_path,'r',encoding='gbk'))
        csv_data=[]
        for stu in csv_file:
            csv_data.append(stu)
    return data,csv_data

def get_data_analysis(predict,label):#get the accuracy of train data predict and label
    error = 0
    Eudi = 0
    predict_ ={}
    label_= {}
    for name in predict:
        predict_[name[0]] = np.array(name[1:],dtype=np.float)
    for name in label:
        label_[name[0]] = np.array(name[1:],dtype=np.float)
    for name in predict_.keys():
        error_ = 0
        # print(predict_[name][0])
        for i in range(3):
            error_ += abs(pow(predict_[name][i],2)-pow(label_[name][i],2))
        error += pow(error_,0.5)
    Eudi += error/len(predict)
    return Eudi


def train(train_data_path, label_path, save_path, scene):#get the transform_matrix


    predict_,label_ = read_file(sfm_data_path=train_data_path,label_path=label_path)
    ####################data_process##################################
    label_ = label_[1:]     # scene truth
    data_train={}
    for img in label_:
        try:
            data_train[img[0]+'.jpg'] = predict_[img[0]+'.jpg']#get the train dataset
        except:
            # print('img {} not in real scene'.format(img[0]))#some test dataset
            pass

    label = []
    predict = []
    name=[]

    for i in range(len(label_)):
        if (label_[i][0]+'.jpg') in data_train:
            a=np.array(label_[i][1:],dtype=np.float)
            b=np.array(predict_[label_[i][0]+'.jpg']['center'],dtype=np.float)
            label.append(a)
            predict.append(b)
            name.append(label_[i][0])

    label=np.array(label)
    predict=np.array(predict)
    data_output =[]
    for i in data_train:
        data_output.append(data_train[i]['center'])


    #######################Affine transformation#######################
    s, A, b=get_transform(predict,label)


    matrix=[]

    data_name = data_train.keys()
    for i in data_name:
        vector =np.array(data_train[i]['center'],dtype=np.float)
        matrix.append(vector)
    matrix = np.array(matrix)

    output = s*A.dot(matrix.T).T+b

    name = []


    for index, nameline in enumerate(data_name):
        name.append(nameline.split('.')[0])



    ##########################caculate_distance###########################

    ana_predict = np.column_stack((name,output))
    label_ = np.array(label_)
    print('The accuracy of scene %d is %f\n' % (scene, get_data_analysis(ana_predict, label_)))

    return  s,A,b

def test_output(test_path,label_path,save_path,s,A,b):
    test_data_ ,_ = read_file(sfm_data_path = test_path,label_path = label_path)
    data_name = test_data_.keys()
    matrix =[]
    for i in data_name:
        vector =np.array(test_data_[i]['center'],dtype=np.float)
        matrix.append(vector)
    matrix = np.array(matrix)

    output = s*A.dot(matrix.T).T+b


    ##########################output_save###########################
    name = []
    new_X =[]
    new_Y = []
    new_Z =[]

    for index,nameline in enumerate(data_name):
        name.append(nameline.split('.')[0])
        new_X.append(output[index][0])
        new_Y.append(output[index][1])
        new_Z.append(output[index][2])
    output = np.array(output,dtype=np.float)
    output = np.column_stack((name,output))
    f = open(save_path,'w')
    f.write('\tname\t\t\tlabel_X\t\tlabel_Y\t\tlabel_Z\n')
    for i in range(len(name)):
        f.write(name[i]+'\t'+str('%.4f'%new_X[i])+'\t'+str('%.4f'%new_Y[i])+'\t'+str('%.4f'%new_Z[i])+'\n')
    f.close


    return output
    # print(test_data_)


def main():
    label_path = ['/home/ncepu-lk/smartcityData/source_pano/scene1/scene1_jiading_lib_training_coordinates',
                  '/home/ncepu-lk/smartcityData/source_pano/scene2/scene2_siping_lib_training_coordinates',
                  '/home/ncepu-lk/smartcityData/source_pano/scene3/scene3_siping_zhonghe_training_coordinates',
                  '/home/ncepu-lk/smartcityData/source_pano/scene4/jiading_zhixin_training_coordinates',
                  '/home/ncepu-lk/smartcityData/source_pano/scene5/jiading_rainbow__training_coordinates',
                  '/home/ncepu-lk/smartcityData/source_pano/scene6/jiading_bolou_training_coordinates',
                  '/home/ncepu-lk/smartcityData/source_pano/scene7/jiading_riverside_training_coordinates',
                  '/home/ncepu-lk/smartcityData/source_pano/scene8/jiading_hualou_training_coordinates']



    predict_data_test =[]
    predict_data_train =[]
    for index, labelpath in enumerate(label_path):
        train_path = '/home/ncepu-lk/smartcityData/results/scene' + str(index + 1) + '/reconstruction/sfm_data.json'
        save_path = '/home/ncepu-lk/smartcityData/results/scene' + str(index + 1) + '.txt'
        train_data_path = '/home/ncepu-lk/smartcityData/results_train/scene' + str(index + 1) + '/reconstruction/sfm_data.json'
        test_data_path = '/home/ncepu-lk/smartcityData/results_test/scene' + str(index + 1) + '/reconstruction/sfm_data.json'
        train_save_path = '/home/ncepu-lk/smartcityData/results_train/scene' + str(index + 1) + '.txt'
        test_save_path = '/home/ncepu-lk/smartcityData/results_test/scene' + str(index + 1) + '.txt'

        s, A, b = train(train_path, labelpath + '.csv', save_path, index)
        predict_data_test.extend(test_output(test_data_path, labelpath + '.csv', test_save_path, s, A, b))
        predict_data_train.extend(test_output(train_data_path, labelpath + '.csv', train_save_path, s, A, b))

    with open('/home/ncepu-lk/smartcityData/results_test/output_test.csv','w') as f:
        w = csv.writer(f)
        w.writerow(('name','X_label','Y_label','Z_label'))
        for row in predict_data_test:
            w.writerow(row)
        f.close()
    with open('/home/ncepu-lk/smartcityData/results_train/output_train.csv', 'w') as f:
        w = csv.writer(f)
        w.writerow(('name', 'X_label', 'Y_label', 'Z_label'))
        for row in predict_data_train:
            w.writerow(row)
        f.close()


score_path = './results/score.txt'
if __name__ == '__main__':
    import os
    if os.path.exists(score_path):
        os.system('rm '+score_path)
    main()






