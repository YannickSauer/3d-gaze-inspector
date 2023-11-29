import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

import matplotlib.pyplot as plt
import matplotlib
import scipy.stats as st



def read_measurement(path,tracker=None,column_names=None):
    if tracker is None:
        tracker = "custom"

    if tracker == "custom": # if custom tracker, then just take the column names passed in the function
        if column_names is None: # if no column names passed, then take the Unity ones
            tracker == "Unity"

    if tracker == "Unity":
        column_names = ['Timestamp', 'left_origin', 'left_gaze', 'right_origin', 'right_gaze', 'comb_origin', 'comb_gaze', 'cam_pos', 'cam_rot', 'pof', 'ooi', 'msg']
    if tracker == "PupilLabs":
        column_names = ['gaze_time', 'eye_center0_3d_x','eye_center0_3d_y','eye_center0_3d_z','gaze_normal0_x','gaze_normalo_y','gaze_normalo_z','eye_center1_3d_x','eye_center1_3d_y','eye_center1_3d_z','gaze_normal1_x','gaze_normal1_y','gaze_normal1_z']

    data = pd.read_csv(path,names=column_names, header=0, index_col=False, delimiter=';')
    if data.empty:
        print('Empty data set:' + str(path))
        return None
    data[['left_orig_x', 'left_orig_y', 'left_orig_z']] = data.left_origin.str.replace(')', '',regex=True) \
        .str.replace('(','',regex=True).str.split(',', expand=True).astype(float)
    data[['left_gaze_x', 'left_gaze_y', 'left_gaze_z']] = data.left_gaze.str.replace(')', '',regex=True) \
        .str.replace('(','',regex=True).str.split(',', expand=True).astype(float)

    data[['right_orig_x', 'right_orig_y', 'right_orig_z']] = data.right_origin.str.replace(')', '',regex=True) \
        .str.replace('(','',regex=True).str.split(',', expand=True).astype(float)
    data[['right_gaze_x', 'right_gaze_y', 'right_gaze_z']] = data.right_gaze.str.replace(')', '',regex=True) \
        .str.replace('(','',regex=True).str.split(',', expand=True).astype(float)

    data[['comb_orig_x','comb_orig_y','comb_orig_z']] = data.comb_origin.str.replace(')', '',regex=True) \
        .str.replace('(','',regex=True).str.split(',', expand=True).astype(float)
    data[['comb_gaze_x','comb_gaze_y','comb_gaze_z']] = data.comb_gaze.str.replace(')', '',regex=True) \
        .str.replace('(','',regex=True).str.split(',', expand=True).astype(float)

    data[['cam_pos_x','cam_pos_y','cam_pos_z']] = data.cam_pos.str.replace(')', '',regex=True) \
        .str.replace('(','',regex=True).str.split(',',expand=True).astype(float)
    data[['cam_rot_x','cam_rot_y','cam_rot_z','cam_rot_w'] ] = data.cam_rot.str.replace(')', '',regex=True) \
        .str.replace('(', '',regex=True).str.split(',',expand=True).astype(float)
    data[['pof_x','pof_y','pof_z']] = data.pof.str.replace(')','',regex=True).str.replace('(', '',regex=True) \
        .str.split(',', expand=True).astype(float)
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

# use the msg collumn to split the tracking data into individual trials
def split_trials(data,start_str,stop_str=None):
    messages = data.msg[~data.msg.isnull()]
    start_ind = messages[messages.str.contains(start_str)].index
    n_trials = len(start_ind)
    stop_ind = []
    if stop_str is not None:
        stop_ind = messages[messages.str.contains(stop_str)].index
        if len(stop_ind) != len(start_ind):
            print("No equal number of start and stop indices.")

    data['trial']=np.nan
    for trial,i in enumerate(start_ind):
        if stop_str is None:
            if (i == n_trials-1):
                data.trial[start_ind[trial]:] = trial
            else:
                data.trial[start_ind[trial]:start_ind[trial + 1]] = trial
        else:
            if (i == n_trials-1) and len(stop_ind) == len(start_ind) - 1: # probably stop marker for last trial missing
                data.trial[start_ind[trial]:] = trial
            else:
                data.trial[start_ind[trial]:stop_ind[trial]] = trial
    return data

def vec_str2array(vecstr):
    arraylist = []
    for string in vecstr:
        arraylist.append(np.fromstring(string.replace('(', '').replace(')', ''), sep=','))
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

def _quantile_to_level(data, quantile):  # stolen from seaborn
    """Return data levels corresponding to quantile cuts of mass."""
    isoprop = np.asarray(quantile)
    values = np.ravel(data)
    sorted_values = np.sort(values)[::-1]
    normalized_values = np.cumsum(sorted_values) / values.sum()
    idx = np.searchsorted(normalized_values, 1 - isoprop)
    levels = np.take(sorted_values, idx, mode="clip")
    return levels

def vergence_point(p1,v1,p2,v2):
    from numpy.linalg import norm, solve
    v1 /= np.linalg.norm(v1,axis=1)[:,np.newaxis]
    v2 /= np.linalg.norm(v1, axis=1)[:, np.newaxis]

    dot_product = np.sum(v1 * v2 ,axis=1)
    angle = np.degrees(np.arccos(dot_product))
    cot_ver_angle = 1/np.tan(np.radians(angle/2))

    result = np.zeros(v1.shape)
    for row in np.arange(v1.shape[0]):
        p1row = p1[row,:]
        p2row = p2[row,:]
        v1row = v1[row,:]
        v2row = v2[row,:]

        # find the unit direction vector of the connecting line
        v3 = np.cross(v2row,v1row,axisa=0,axisb=0)
        v3 /= norm(v3,axis=0)
        RHS = p2row - p1row
        LHS = np.array([v1, -v2, v3]).T
        t = solve(LHS, RHS)
        result[row,:] = p1row + t[0]*v1 + t[2]/2*v3

    return result , np.linalg.norm(result,axis=1)

def get_longlat_kde(long,lat,long_sample,lat_sample,bandwidth = np.radians(2)):
    from sklearn.neighbors import KernelDensity


    # prepare training data by removing NaNs
    ind = ~np.isnan(long)
    x = long[ind]
    y = lat[ind]
    ind = ~np.isnan(y)
    x = x[ind]
    y = y[ind]
    xtrain = np.vstack([x,y]).T

    # train Kernel Density Estimator
    kde = KernelDensity(metric='haversine', bandwidth=bandwidth)
    kde.fit(xtrain)

    # evaluate KDE at sampling points
    xsampl = np.vstack([long_sample.ravel(),lat_sample.ravel()]).T
    img_kde = np.exp(kde.score_samples(xsampl))
    img_kde = img_kde.reshape(long_sample.shape)
    return img_kde

# calculate the solid angle covered by the gaze heatmap with the specified quantile
# long,lat in radians
def solid_angle_kde(long,lat, # recorded gaze points
                    dlong=np.radians(1),dlat=np.radians(1), # stepsize of sampling points
                    quantile = 0.05): # defines threshold used to calculate "gaze area"

    long_samp,lat_samp  = np.mgrid[slice(-np.radians(60),np.radians(60),dlong),
                                   slice(-np.radians(42),np.radians(40),dlat)]

    kde = get_longlat_kde(long,lat,long_samp,lat_samp)

    # calculate solid angle element for each "pixel" of the kde
    dOmega = np.cos(lat_samp) * dlong * dlat
    # get treshold level for quantile mass in kde
    threshold = _quantile_to_level(kde, quantile)
    omega = (dOmega * (kde>=threshold)).sum() # sum all dOmega elements where distribution >= threshold
    return omega , kde


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