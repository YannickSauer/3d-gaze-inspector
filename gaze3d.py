import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

import matplotlib.pyplot as plt
import matplotlib
import scipy.stats as st


def read_measurement(path):
    names = ['Timestamp','left_origin','left_gaze','right_origin','right_gaze','comb_origin','comb_gaze','cam_pos','cam_rot','pof','ooi','msg']
    data = pd.read_csv(path,names=names,header=0,index_col=False,delimiter=';')
    if data.empty:
        print('Empty data set:' + path)
        return None
    data[['left_orig_x','left_orig_y','left_orig_z']]=data.left_origin.str.replace(')','').str.replace('(','').str.split(',',expand=True).astype(float)
    data[['left_gaze_x','left_gaze_y','left_gaze_z']]=data.left_gaze.str.replace(')','').str.replace('(','').str.split(',',expand=True).astype(float)
    data[['right_orig_x','right_orig_y','right_orig_z']]=data.right_origin.str.replace(')','').str.replace('(','').str.split(',',expand=True).astype(float)
    data[['right_gaze_x','right_gaze_y','right_gaze_z']]=data.right_gaze.str.replace(')','').str.replace('(','').str.split(',',expand=True).astype(float)
    
    data[['comb_orig_x','comb_orig_y','comb_orig_z']]=data.comb_origin.str.replace(')','').str.replace('(','').str.split(',',expand=True).astype(float)
    data[['comb_gaze_x','comb_gaze_y','comb_gaze_z']]=data.comb_gaze.str.replace(')','').str.replace('(','').str.split(',',expand=True).astype(float)
    
    data[['cam_pos_x','cam_pos_y','cam_pos_z']] = data.cam_pos.str.replace(')','').str.replace('(','').str.split(',',expand=True).astype(float)
    data[['cam_rot_x','cam_rot_y','cam_rot_z','cam_rot_w'] ]= data.cam_rot.str.replace(')','').str.replace('(','').str.split(',',expand=True).astype(float)
    data[['pof_x','pof_y','pof_z']] = data.pof.str.replace(')','').str.replace('(','').str.split(',',expand=True).astype(float)
    #data = data.drop(columns=['left_origin', 'left_gaze','right_origin','right_gaze','comb_origin','comb_gaze','cam_pos','cam_rot'])
    
    data.left_origin = vec_str2array(data.left_origin)
    data.left_gaze = vec_str2array(data.left_gaze)
    data.right_origin = vec_str2array(data.right_origin)
    data.right_gaze = vec_str2array(data.right_gaze)
    data.comb_origin = vec_str2array(data.comb_origin)
    data.comb_gaze = vec_str2array(data.comb_gaze)
    data.cam_pos = vec_str2array(data.cam_pos)
    data.cam_rot = vec_str2array(data.cam_rot)
    return data   

def vec_str2array(vecstr):
    arraylist = []
    for string in vecstr:
        arraylist.append(np.fromstring(string.replace('(','').replace(')',''),sep=','))
    return arraylist

# convert (x,y,z) to spherical coordinates with eccentricity theta and meridian phi 
def cart2spherical(x,y,z):
    xy_sqr = x**2 + y**2
    r = np.sqrt(xy_sqr + z**2)
    theta = np.arctan2(np.sqrt(xy_sqr),z)
    phi = np.arctan2(y, x)
    return [r,theta,phi]

# convert (x,y,z) to longitude and latitude coordinates
def cart2geographic(x,y,z):
    r = np.sqrt(x**2 + y**2 + z**2)
    # z axis is forward
    # longitude is angle in the x-z-plane
    longitude = np.arctan2(x,z)
    latitude = np.arcsin(y/r) 
    return r,longitude,latitude

def rotate_vec(xyz,quaternion):
    r = Rotation.from_quat(quaternion)
    return r.apply(xyz)

def world_path_plot(data):
    """ Creates a plot of the cameras path in 3d with heatmap on the ground"""
    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d')
    x = data['cam_pos_x']
    y = data['cam_pos_y']
    z = data['cam_pos_z']
    ax.set_xlabel('x (m)')
    ax.set_ylabel('z (m)')
    ax.set_zlabel('y (m)')
   
    xmin, xmax = -3, 3
    ymin, ymax = -3, 3

    # Peform the kernel density estimate
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    kernel = st.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)
    # Contourf plot
    cfset = ax.contourf(xx, yy, f, cmap='Blues')
    ax.plot(x,z,y)
    ax.plot(x,z,0)
    
def quat2ypr(quaternion):
    r = Rotation.from_quat(quaternion)
    ypr = r.as_euler('YXZ', degrees=True)
    return ypr[:,0],ypr[:,1],ypr[:,2]
    
def vergence_distance(left_orig,left_gaze,right_orig,right_gaze):
    iod = np.sqrt(right_orig**2 - left_orig**2)
    # angle between left and right gaze
    return iod

def head_angular_vel(head_quaternion,t):
    r = Rotation.from_quat(head_quaternion)
    diff = r[1:] * r[:-1].inv() # get the rotations between r[n] and r[n+1]
    dphi = diff.magnitude()
    dt = t[1:] - t[:-1]
    return dphi/dt

def unit_vector(vec):
    """ Returns the unit vector of the vector.  """
    return vec / np.linalg.norm(vec)

def angle_between(v1, v2):
    """ Returns the angle in deg between vectors 'v1' and 'v2' """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.rad2deg(np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0)))

def gaze_angular_vel(gaze_vec,t):
    """ Returns the angular velocity of gaze_vec at time t"""
    d_phi = angle_between(gaze_vec[1:],gaze_vec[:-1])
    d_t = t[1:] - t[:-1]
    return d_phi/d_t
    
    
def get_trial_index(data):
    # get all non NaN cells:
    messages = data.msg[~data.msg.isnull()]
    ind = messages.index
    start_index = np.empty((330,3))
    end_index = np.empty((330,3))
    start_index[:] = np.nan
    end_index[:] = np.nan
    for i in range(len(messages)):
        trial = int(messages.iloc[i].lower().split("trial")[1].split("stim")[0])
        stiml = int(messages.iloc[i].lower().split("trial")[1].split("stim")[1])
        if('start' in messages.iloc[i].lower()):
            start_index[trial,stiml] = ind[i]
        if('end' in messages.iloc[i].lower()):
            stop_index[trial,stiml] = ind[i]
    return start_index , stop_index
    