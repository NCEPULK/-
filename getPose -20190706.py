import json
import csv
import numpy
import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pdb

def quaternion_matrix(quaternion):
    # Return homogeneous rotation matrix from quaternion.
    q = numpy.array(quaternion, dtype=numpy.float64, copy=True)
    n = numpy.dot(q, q)
    if n < _EPS:
        return numpy.identity(4)
    q *= math.sqrt(2.0 / n)
    q = numpy.outer(q, q)
    return numpy.array([
        [1.0-q[2, 2]-q[3, 3],     q[1, 2]-q[3, 0],     q[1, 3]+q[2, 0], 0.0],
        [    q[1, 2]+q[3, 0], 1.0-q[1, 1]-q[3, 3],     q[2, 3]-q[1, 0], 0.0],
        [    q[1, 3]-q[2, 0],     q[2, 3]+q[1, 0], 1.0-q[1, 1]-q[2, 2], 0.0],
        [                0.0,                 0.0,                 0.0, 1.0]])

def affine_matrix_from_points(v0, v1, shear=True, scale=True, usesvd=True):
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

    if shear:
        # Affine transformation
        A = numpy.concatenate((v0, v1), axis=0)#concatenate data vo v1
        u, s, vh = numpy.linalg.svd(A.T)#Singular value decomposition A.T = u*s*vh
        vh = vh[:ndims].T
        B = vh[:ndims]
        C = vh[ndims:2*ndims]
        t = numpy.dot(C, numpy.linalg.pinv(B))
        t = numpy.concatenate((t, numpy.zeros((ndims, 1))), axis=1)
        M = numpy.vstack((t, ((0.0,)*ndims) + (1.0,)))


    elif usesvd or ndims != 3:
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
    else:
        # Rigid transformation matrix via quaternion
        # compute symmetric matrix N
        xx, yy, zz = numpy.sum(v0 * v1, axis=1)
        xy, yz, zx = numpy.sum(v0 * numpy.roll(v1, -1, axis=0), axis=1)
        xz, yx, zy = numpy.sum(v0 * numpy.roll(v1, -2, axis=0), axis=1)
        N = [[xx+yy+zz, 0.0,      0.0,      0.0],
             [yz-zy,    xx-yy-zz, 0.0,      0.0],
             [zx-xz,    xy+yx,    yy-xx-zz, 0.0],
             [xy-yx,    zx+xz,    yz+zy,    zz-xx-yy]]
        # quaternion: eigenvector corresponding to most positive eigenvalue
        w, V = numpy.linalg.eigh(N)
        q = V[:, numpy.argmax(w)]
        q /= vector_norm(q)  # unit quaternion
        # homogeneous transformation matrix
        M = quaternion_matrix(q)

    if scale and not shear:
        # Affine transformation; scale is ratio of RMS deviations from centroid
        v0 *= v0
        v1 *= v1
        M[:ndims, :ndims] *= math.sqrt(numpy.sum(v1) / numpy.sum(v0))

    # move centroids back
    M = numpy.dot(numpy.linalg.inv(M1), numpy.dot(M, M0))
    M /= M[ndims, ndims]
    return M

def superimposition_matrix(v0, v1, scale=False, usesvd=True):
    # Return matrix to transform given 3D point set into second point set.
    v0 = numpy.array(v0, dtype=numpy.float64, copy=False)[:3]
    v1 = numpy.array(v1, dtype=numpy.float64, copy=False)[:3]
    return affine_matrix_from_points(v0, v1, shear=False,
                                     scale=scale, usesvd=usesvd)

def align_reconstruction_naive_similarity(X, Xp):
    """Align with GPS and GCP data using direct 3D-3D matches."""
    # Compute similarity Xp = s A X + b

    T = superimposition_matrix(X.T, Xp.T, scale=True)
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

        # not all extrinsics has [img['key']]
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

def get_data_analysis(predict,label):
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

def test_output(test_path):
    test_data_ ,_ = read_file(sfm_data_path = test_path,label_path = label_path)


def main(sfm_data_path, label_path, save_path, scene):


    predict_,label_ = read_file(sfm_data_path=sfm_data_path,label_path=label_path)
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
        #pdb.set_trace()
        data_output.append(data_train[i]['center'])
    data_output = np.array(data_output)

    #######################Affine transformation#######################
    s, A, b=align_reconstruction_naive_similarity(predict,label)


    cc=[]

    data_name = data_train.keys()
    for i in data_name:
        c =np.array(data_train[i]['center'],dtype=np.float)
        cc.append(c)

    cc=np.array(cc)

    new_c=s*A.dot(cc.T).T+b


    ##########################output_save###########################
    name = []
    new_X =[]
    new_Y = []
    new_Z =[]

    for index,nameline in enumerate(data_name):
        name.append(nameline.split('.')[0])
        new_X.append(new_c[index][0])
        new_Y.append(new_c[index][1])
        new_Z.append(new_c[index][2])

    f = open(save_path,'w')
    f.write('\tname\t\t\tlabel_X\t\tlabel_Y\t\tlabel_Z\n')
    for i in range(len(name)):
        f.write(name[i]+'\t'+str('%.4f'%new_X[i])+'\t'+str('%.4f'%new_Y[i])+'\t'+str('%.4f'%new_Z[i])+'\n')
    f.close

    ##########################caculate_distance###########################

    ana_predict = np.column_stack((name,new_X,new_Y,new_Z))
    label_ = np.array(label_)
    print('The accuracy of scene is %f\n' % ( get_data_analysis(ana_predict, label_)))



    return  name,new_X,new_Y,new_Z


score_path = './results/score.txt'
if __name__ == '__main__':
    import os
    if os.path.exists(score_path):
        os.system('rm '+score_path)


    label_path = ['/home/ncepu-lk/smartcityData/source_pano/scene1/scene1_jiading_lib_training_coordinates',
     '/home/ncepu-lk/smartcityData/source_pano/scene2/scene2_siping_lib_training_coordinates',
     '/home/ncepu-lk/smartcityData/source_pano/scene3/scene3_siping_zhonghe_training_coordinates',
     '/home/ncepu-lk/smartcityData/source_pano/scene4/jiading_zhixin_training_coordinates',
     '/home/ncepu-lk/smartcityData/source_pano/scene5/jiading_rainbow__training_coordinates',
     '/home/ncepu-lk/smartcityData/source_pano/scene6/jiading_bolou_training_coordinates',
     '/home/ncepu-lk/smartcityData/source_pano/scene7/jiading_riverside_training_coordinates',
     '/home/ncepu-lk/smartcityData/source_pano/scene8/jiading_hualou_training_coordinates']

    list = np.array(range(1,9))
    name =[]
    X_label =[]
    Y_label =[]
    Z_label =[]
    for index,labelpath in enumerate(label_path):
        # try:
        sfm_data_path = '/home/ncepu-lk/smartcityData/results/scene'+str(index+1)+'/reconstruction/sfm_data.json'


        save_path = '/home/ncepu-lk/smartcityData/results/scene'+str(index+1)+'.txt'
        name_,X_label_,Y_label_,Z_label_= main(sfm_data_path, labelpath+'.csv', save_path, index)
        name.extend(name_)
        X_label.extend(X_label_)
        Y_label.extend(Y_label_)
        Z_label.extend(Z_label_)
        # print(len(name_))
    output = np.row_stack((name, X_label, Y_label, Z_label)).T
    with open('/home/ncepu-lk/smartcityData/results/output.csv','w') as f:
        w = csv.writer(f)
        w.writerow(('name','X_label','Y_label','Z_label'))
        for row in output:
            w.writerow(row)
        f.close()


