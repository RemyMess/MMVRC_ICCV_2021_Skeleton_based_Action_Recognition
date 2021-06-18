"""
Utility script containing pre-processing logic.
Main call is pre_normalization: 
    Pads empty frames, 
    Centers human, 
    Align joints to axes.
"""

import math

import os
import argparse

import numpy as np
import pickle
from tqdm import tqdm
from tool.validation_tools import switchPeople
from tool.validation_tools import findEffectiveLength
from tool.validation_tools import eliminate_spikes
from scipy.signal import savgol_filter


def pre_normalization(data, center_mass_joints = [5,6,11,12], yaxis=[11, 5], xaxis=[]):
    """
    Normalization steps:
        1) Switch bodies, if it reduces the distance between frames
        2) Center the human at origin
        3) Rotate human to align specified joints to y-axis: ntu [0,1], uav [11,5], aligns to x-axis, if an x-axis argument is given and yaxis = none.
        4) Pads missing frames at the end with last frame
    
    Args:
        data: tensor with skeleton data of shape N x C x T x V x M
        center_joint: list of body joint indexes, the center of mass is the mean between those joints
        yaxis: list containing 0 or 2 body joint indices (0 skips the alignment)
        xaxis: list containing 0 or 2 body joint indices (0 skips the alignment)
    """

    switch_bodies = True
    centering = True
    rotating = True
    scaling = True
    eliminating_spikes = True
    padding_frames = True
    smoothen = False

    N, C, T, V, M = data.shape

    if switch_bodies:
        print('Switch bodies')
        for i in tqdm(range(N)):
            data[i,:,:,:,:] = switchPeople(data[i,:,:,:,:])
    
    s = np.transpose(data, [0, 4, 2, 3, 1])  # N, C, T, V, M  to  N, M, T, V, C

    if centering:
        print('sub the center of mass (mean between the shoulder and hip joints))')
        for i_s, skeleton in enumerate(tqdm(s)):
            if skeleton.sum() == 0:
                continue
            joints = []
            for i in range(len(center_mass_joints)):
                joints.append(skeleton[0][:, center_mass_joints[i]:center_mass_joints[i]+1, :].copy())
            joints = np.array(joints)
            main_body_center = np.mean(joints,axis = 0)
            for i_p, person in enumerate(skeleton):
                if person.sum() == 0:
                    continue
                mask = (person.sum(-1) != 0).reshape(T, V, 1)
                s[i_s, i_p] = (s[i_s, i_p] - main_body_center) * mask

    def align_human_to_vector(joint_idx1: int, joint_idx2: int, target_vector: list):
        for i_s, skeleton in enumerate(tqdm(s)):
            if skeleton.sum() == 0:
                continue
            joint1 = skeleton[0, 0, joint_idx1]
            joint2 = skeleton[0, 0, joint_idx2]
            axis = np.cross(joint2 - joint1, target_vector)
            angle = angle_between(joint2 - joint1, target_vector)
            matrix = rotation_matrix(axis, angle)
            for i_p, person in enumerate(skeleton):
                if person.sum() == 0:
                    continue
                for i_f, frame in enumerate(person):
                    if frame.sum() == 0:
                        continue
                    for i_j, joint in enumerate(frame):
                        s[i_s, i_p, i_f, i_j] = np.dot(matrix, joint)
    if rotating:
        if yaxis:
            print('parallel the bone between hip(jpt %s)' %yaxis[0] + \
                'and spine(jpt %s) of the first person to the y axis' %yaxis[1])
            align_human_to_vector(yaxis[0], yaxis[1], [0, 1, 0])
        if xaxis:
            print('parallel the bone between right shoulder(jpt %s)' %xaxis[0] + \
                'and left shoulder(jpt %s) of the first person to the x axis' %xaxis[1])
            align_human_to_vector(xaxis[0], xaxis[1], [1, 0, 0])

    data = np.transpose(s, [0, 4, 2, 3, 1])

    def scaleData(data):
        print('Rescaling the data')
        for i in tqdm(range(N)):
            m = find_mean_dist(data[i,:,:,:,:])
            if m == 0:
                continue
            factor = 1/m
            data[i,:,:,:,:] = factor * data[i,:,:,:,:]
        return data
    
    if scaling:
        data = scaleData(data)

    if eliminating_spikes:
        print('Eliminate spikes')
        for i in tqdm(range(N)):
            for m in range(M):
                if data[i,:,:,:,m].sum() == 0:
                    continue
                data[i,:,:,:,m] = eliminate_spikes(data[i,:,:,:,m])
            
    if scaling:
        data = scaleData(data)
        
    if padding_frames:
        print('pad missing frames with last frame')
        for i in tqdm(range(N)):
            data[i,:,:,:,:] = padFrames(data[i,:,:,:,:])

    print('delete unnecessary info')
    data = np.delete(data, obj = 2, axis = 1)
    data = np.delete(data, obj = range(305,T), axis = 2)

    if smoothen:
        print('apply Savgol filter')
        data = savgol_filter(data, 5, 2, axis = 2)
    
    return data

def rotation_matrix(axis, theta):
    """
    Return the rotation matrix associated with counterclockwise rotation about
    the given axis by theta radians.
    """
    if np.abs(axis).sum() < 1e-6 or np.abs(theta) < 1e-6:
        return np.eye(3)
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def padFrames(sample):
    """fills empty frames at the end with the last non-empty frame"""
    C,T,V,M = sample.shape
    effective_length = findEffectiveLength(sample)

    for i in range(effective_length,T):
        sample[:,i,:,:] = sample[:,effective_length-1,:,:]
    return sample


def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    if np.abs(v1).sum() < 1e-6 or np.abs(v2).sum() < 1e-6:
        return 0
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))


def x_rotation(vector, theta):
    """Rotates 3-D vector around x-axis"""
    R = np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])
    return np.dot(R, vector)


def y_rotation(vector, theta):
    """Rotates 3-D vector around y-axis"""
    R = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
    return np.dot(R, vector)


def z_rotation(vector, theta):
    """Rotates 3-D vector around z-axis"""
    R = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    return np.dot(R, vector)

def find_mean_dist(sample, joints = [6,11]):
    """
    calculates the mean distance between the two joints of the first person over all frames of a sample, ignoring 0-frames
    Default joints: right shoulder, left hip
    """
    C,T,V,M = sample.shape
    effective_length = findEffectiveLength(sample)

    joint_1 = sample[:,:,joints[0],0]
    joint_2 = sample[:,:,joints[1],0]

    dist = np.linalg.norm(joint_1-joint_2, axis = 0)

    return np.sum(dist)/effective_length

def load_uav_data(dir_name, flag = 'train'):

    filename = os.path.join(dir_name,'{}_raw_data.npy'.format(flag))
    data_uav = np.load(filename, mmap_mode = None)
    N,C,T,V,M = data_uav.shape
    #print(N,C,T,V,M)

    with open(os.path.join(dir_name,'{}_label.pkl'.format(flag)), 'rb') as f:
        sample_name, label_uav = pickle.load(f)
    
    label_uav = np.array(label_uav)
    #print(label_uav.shape)
    return data_uav,label_uav,np.array(sample_name)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='UAVHuman Data Converter.')
    parser.add_argument('--data_path', required=True)
    args = parser.parse_args()

    DATASET_PATH = args.data_path
    isSaveProcessData = True

    for flag in ['train']:

        data_uav, label_uav, sample_name = load_uav_data(DATASET_PATH, flag)
        print('raw_data shape: ', data_uav.shape)

        process_data = pre_normalization(data_uav)
        print('process_data shape: ', process_data.shape)

        if isSaveProcessData:
            np.save(os.path.join(DATASET_PATH,'{}_process_data.npy'.format(flag)), process_data)
    
    print('............End............. ')
