import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

import matplotlib.pyplot as plt
import matplotlib

def read_measurement(path):
    names = ['Timestamp','left_origin','left_gaze','right_origin','right_gaze','comb_origin','comb_gaze','cam_pos','cam_rot','msg']
    data = pd.read_csv(path,names=names,header=0,index_col=False,delimiter=';')
    data[['left_orig_x','left_orig_y','left_orig_z']]=data.left_origin.str.replace(')','').str.replace('(','').str.split(',',expand=True).astype(float)
    data[['left_gaze_x','left_gaze_y','left_gaze_z']]=data.left_gaze.str.replace(')','').str.replace('(','').str.split(',',expand=True).astype(float)
    data[['right_orig_x','right_orig_y','right_orig_z']]=data.right_origin.str.replace(')','').str.replace('(','').str.split(',',expand=True).astype(float)
    data[['right_gaze_x','right_gaze_y','right_gaze_z']]=data.right_gaze.str.replace(')','').str.replace('(','').str.split(',',expand=True).astype(float)
    
    data[['comb_orig_x','comb_orig_y','comb_orig_z']]=data.comb_origin.str.replace(')','').str.replace('(','').str.split(',',expand=True).astype(float)
    data[['comb_gaze_x','comb_gaze_y','comb_gaze_z']]=data.comb_gaze.str.replace(')','').str.replace('(','').str.split(',',expand=True).astype(float)
    
    data[['cam_pos_x','cam_pos_y','cam_pos_z']]=data.cam_pos.str.replace(')','').str.replace('(','').str.split(',',expand=True).astype(float)
    data[['cam_rot_x','cam_rot_y','cam_rot_z','cam_rot_w']]=data.cam_rot.str.replace(')','').str.replace('(','').str.split(',',expand=True).astype(float)
    
    #data = data.drop(columns=['left_origin', 'left_gaze','right_origin','right_gaze','comb_origin','comb_gaze','cam_pos','cam_rot'])
    
    data.left_origin = vecStr2array(data.left_origin)
    data.left_gaze = vecStr2array(data.left_gaze)
    data.right_origin = vecStr2array(data.right_origin)
    data.right_gaze = vecStr2array(data.right_gaze)
    data.comb_origin = vecStr2array(data.comb_origin)
    data.comb_gaze = vecStr2array(data.comb_gaze)
    data.cam_pos = vecStr2array(data.cam_pos)
    data.cam_rot = vecStr2array(data.cam_rot)
    return data
    

def vecStr2array(vecstr):
    arraylist = []
    for string in vecstr:
        arraylist.append(np.fromstring(string.replace('(','').replace(')',''),sep=','))
    return arraylist

def cart2spherical(x,y,z):
    xy_sqr = x**2 + y**2
    r = np.sqrt(xy_sqr + z**2)
    theta = np.arctan2(np.sqrt(xy_sqr),z)
    phi = np.arctan2(y, x)
    return [r,theta,phi]

def rotateVec(xyz,quaternion):
    r = Rotation.from_quat(quaternion)
    return r.apply(xyz)
    
def quat2ypr(quaternion):
    r = Rotation.from_quat(quaternion)
    ypr = r.as_euler('YXZ', degrees=True)
    return ypr[:,0],ypr[:,1],ypr[:,2]
    
def vergenceDistance(left_orig,left_gaze,right_orig,right_gaze):
    iod = np.sqrt(right_orig**2 - left_orig**2)
    # angle between left and right gaze
    return iod