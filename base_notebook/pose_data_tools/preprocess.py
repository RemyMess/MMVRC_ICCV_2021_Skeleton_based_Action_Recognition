#import sys
#sys.path.extend(['../'])
import os
import numpy as np
from tqdm import tqdm
import math
from joblib import Parallel, delayed
import argparse
import pickle

'''
Joint sequence same as COCO format: {
    0: nose,
    1: left_eye,
    2: right_eye,
    3: left_ear,
    4: right_ear,
    5: left_shoulder,
    6: right_shoulder,
    7: left_elbow,
    8: right_elbow,
    9: left_wrist,
    10: right_wrist,
    11: left_hip,
    12: right_hip,
    13: left_knee,
    14: right_knee,
    15: left_ankle,
    16: right_ankle
}
'''

def pre_normalization(data, ndim = 2, zaxis=[[11,12], [5,6]], xaxis=[[6,11], [5,12]]):#, zaxis=[[11,12], [5,6]], xaxis=[[6,11], [5,12]]

    N, C, T, V, M = data.shape
    if C < ndim:
        raise ValueError('Wrong Data dimension!')

    s = np.transpose(data, [0, 4, 2, 3, 1])  # to (N, M, T, V, C)
    isPadding = True
    isCentering = True
    is3DRotation = True if ndim == 3 else False
    isScaling = True

    isParallel = True

    if isPadding:
        print("Remove zero skeletons on both ends, and then pad the intermedia null frames with the next valid frames")
        
        skeletons = []
        if isParallel:
            skeletons = Parallel(n_jobs=-1)(delayed(parallelPad)(skeleton) for skeleton in tqdm(s[...,:ndim]))
        else:
            for skeleton in tqdm(s[...,:ndim]):  # Dimension N
                skeleton,length = parallelPad(skeleton)
                skeletons.append(skeleton)
        s[...,:ndim] = np.stack(skeletons)

        """
        #print('skip the null frames')
        if person[0].sum() == 0:
            # `index` of frames that have non-zero nodes
            index = (person.sum(-1).sum(-1) != 0)
            tmp = person[index].copy()
            # Shift non-zero nodes to beginning of frames
            person *= 0
            person[:len(tmp)] = tmp

        index_ranges = []
        for i_p, person in enumerate(skeleton):
            # `index` of frames that have non-zero nodes
            index = (person.sum(-1).sum(-1) != 0)
            index_ranges.append([index[0],index[-1]])
        index_ranges = np.array(index_ranges)
        index_start = index_ranges.min(0)[0]
        index_end = index_ranges.max(0)[1]
        tmp = skeleton[:,index_start:index_end].copy()
        # Shift non-zero nodes to beginning of frames
        skeleton *= 0
        skeleton[:,:len(tmp)] = tmp

        for i_f, frame in enumerate(person):
            # Each frame has shape (V, C)
            if frame.sum() == 0:
                if person[i_f:].sum() == 0:
                    # Repeat all the frames up to now (`i_f`) till the max seq len
                    rest = len(person) - i_f
                    reps = int(np.ceil(rest / i_f))
                    pad = np.concatenate([person[:i_f] for _ in range(reps)], 0)[:rest]
                    s[i_s, i_p, i_f:] = pad
                    break
        """ 

    lengths = Parallel(n_jobs=-1)(delayed(parallel_get_duration)(d) for d in np.transpose(s, [0, 2, 1, 3, 4]))
    lengths = np.stack(lengths)
    print("lengths info: ",lengths.mean(),lengths.max(),lengths.min())

    if isCentering:
        print('sub the center joint of the first frame (spine joint in ntu and neck joint in kinetics)')
        index = np.array([5,6,11,12],dtype=np.int64)
        skeletons = []
        if isParallel:
            skeletons = Parallel(n_jobs=-1)(delayed(parallelCentering)(skeleton,index,lengths[i]) for i,skeleton in enumerate(tqdm(s[...,:ndim])))
        else:
            for skeleton in tqdm(s[...,:ndim]):
                skeleton = parallelCentering(skeleton,index)
                skeletons.append(skeleton)
        s[...,:ndim] = np.stack(skeletons)

    if is3DRotation:
        print('parallel the bone between (jpt {}) and (jpt {}) of the person to the z axis'.format(zaxis[0],zaxis[1]))
        print('parallel the bone between right shoulder(jpt {}) and left shoulder(jpt {}) of the person to the x axis'.format(xaxis[0],xaxis[1]))
        zaxis,xaxis = np.array(zaxis,dtype=np.int64), np.array(xaxis,dtype=np.int64)
        skeletons = []
        if isParallel:
            skeletons = Parallel(n_jobs=-1)(delayed(parallel3DRotation)(skeleton,zaxis,xaxis,lengths[i]) for i,skeleton in enumerate(tqdm(s[...,:ndim])))
        else:
            for skeleton in tqdm(s[...,:ndim]):
                skeleton = parallel3DRotation(skeleton,zaxis,xaxis)
                skeletons.append(skeleton)
        
        s[...,:ndim] = np.stack(skeletons)
    


    if isScaling:
        print('rescale each object sequence to the range [0,1] while maintaing the high-width ratio')
        skeletons = []
        if isParallel:
            skeletons = Parallel(n_jobs=-1)(delayed(parallelScale)(skeleton,lengths[i]) for i,skeleton in enumerate(tqdm(s[...,:ndim])))
        else:
            for skeleton in tqdm(s[...,:ndim]):
                skeleton = parallelScale(skeleton)
                skeletons.append(skeleton)
        s[...,:ndim] = np.stack(skeletons)

    data = np.transpose(s, [0, 4, 2, 3, 1])
    return data, lengths


def parallelPad(skeleton):
    if skeleton.sum() == 0:
        print('This sample has no skeleton')
        return skeleton

    # Shift non-zero nodes to beginning of frames by computing the valid range of all persons
    index_ranges = []
    for i_p, person in enumerate(skeleton):
        #if i_p > 0: break # uncomment if we only use the first person to get the valid index range
        # `index` of frames that have non-zero nodes
        index = np.where(person.sum(-1).sum(-1) != 0)[0]
        if len(index) > 0:
            index_ranges.append([index[0],index[-1]])
    if len(index_ranges) > 0:
        index_ranges = np.array(index_ranges)
        index_start = index_ranges.min(0)[0]
        index_end = index_ranges.max(0)[1]
    else:
        raise ValueError('no skeleton')

    tmp = skeleton[:,index_start:index_end+1].copy()
    skeleton *= 0
    length = index_end-index_start+1
    skeleton[:,:length] = tmp


    # pad the null frames with the next valid frames
    for i_p, person in enumerate(skeleton): # Dimension M (# person)
        # `person` has shape (T, V, C)
        if person.sum() == 0:
            continue
        isFirstIn = True
        for i_f, frame in reversed(list(enumerate(person))):# in a reversed order
            if isFirstIn: 
                isFirstIn = False
                continue
            if frame.sum() == 0:
                skeleton[i_p, i_f] = skeleton[i_p, i_f+1]
    
    # pad the ending null frames with the previous valid frames
    for i_p, person in enumerate(skeleton): # Dimension M (# person)
        # `person` has shape (T, V, C)
        if person.sum() == 0:
            continue
        for i_f, frame in enumerate(person):
            if i_f is 0: continue# after prevous step, the first frame should be non-zero and valid now.
            if frame.sum() == 0:
                if i_f < length: skeleton[i_p, i_f] = skeleton[i_p, i_f-1]
                #s[i_s, i_p, i_f] = s[i_s, i_p, i_f-1]
    
    return skeleton

def parallelCentering(skeleton_origin,index,length):
    if skeleton_origin.sum() == 0:
        return skeleton_origin

    skeleton = skeleton_origin[:,:length,...]
    M, T, V, C = skeleton.shape
    # Use the first skeleton's body center (`1:2` along the nodes dimension)
    main_body_center = skeleton[0][:, index, :].mean(1,keepdims=True).copy()    # Shape (T, 4, C) -> Shape (T, 1, C)
    #main_body_center = skeleton[0][:, 1:2, :].copy()    # Shape (T, 1, C)
    #main_body_center = skeleton[0][:1, 1:2, :].copy()    # Shape (1, 1, C)
    for i_p, person in enumerate(skeleton):
        if person.sum() == 0:
            continue
        # For all `person`, compute the `mask` which is the non-zero channel dimension
        mask = (person.sum(-1) != 0).reshape(T, V, 1)
        # Subtract the first skeleton's centre joint, s.shape = (N, M, T, V, C)
        skeleton[i_p, ..., :C] = (skeleton[i_p, ..., :C] - main_body_center) * mask
    skeleton_origin[:,:length,...] = skeleton
    return skeleton_origin

def parallel3DRotation(skeleton_origin,zaxis,xaxis,length):
    if skeleton_origin.sum() == 0:
        return skeleton_origin
    skeleton = skeleton_origin[:,:length,...]
    M, T, V, C = skeleton.shape
    # Shapes: (C,)
    
    for i_p, person in enumerate(skeleton):
        if person.sum() == 0:
            continue
        for i_f, frame in enumerate(person):
            if frame.sum() == 0:
                continue
            
            joint_bottom = skeleton[0, i_f, zaxis[0]].reshape(-1,C).mean(0)
            joint_top = skeleton[0, i_f, zaxis[1]].reshape(-1,C).mean(0)
            axis = np.cross(joint_top - joint_bottom, [0, 0, 1])
            angle = angle_between(joint_top - joint_bottom, [0, 0, 1])
            matrix_z = rotation_matrix(axis, angle)

            joint_rshoulder = skeleton[0, i_f, xaxis[0]].reshape(-1,C).mean(0)
            joint_lshoulder = skeleton[0, i_f, xaxis[1]].reshape(-1,C).mean(0)
            axis = np.cross(joint_rshoulder - joint_lshoulder, [1, 0, 0])
            angle = angle_between(joint_rshoulder - joint_lshoulder, [1, 0, 0])
            matrix_x = rotation_matrix(axis, angle)

            for i_j, joint in enumerate(frame):
                skeleton[i_p, i_f, i_j, :3] = np.dot(np.dot(matrix_x, matrix_z), joint)
    skeleton_origin[:,:length,...] = skeleton
    return skeleton_origin

def parallelScale(skeleton_origin,length,isCenter2Origin=False,isKeepHWRatio=True,scaleFactor=1.0):#(M, T, V, C)
    skeleton = skeleton_origin[0:1,:length,...]
    nDim = skeleton.shape[-1]
    ma = (skeleton.reshape(-1,nDim)).max(axis = 0)
    mi = (skeleton.reshape(-1,nDim)).min(axis = 0)
    skeleton = skeleton_origin[:,:length,...]
    skeleton = (skeleton - mi)*scaleFactor/((ma - mi).max() if isKeepHWRatio else (ma - mi))
    if isCenter2Origin: skeleton -= 0.5*scaleFactor   
    skeleton_origin[:,:length,...] = skeleton 
    return skeleton_origin

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



def parallel_get_duration(sample):#T,M,V,C
    T = sample.shape[0]
    length = T
    for t in range(T-1,-1,-1):
        if np.fabs(sample[t]).sum(-1).sum(-1).sum(-1) > 0.0001:
            length = t + 1
            break
    return length

def get_data_duration(data_uav):#nctvm
    print('Get length info from data...')
    data_uav = np.transpose(data_uav,(0,2,4,3,1))
    N,T,M,V,C = data_uav.shape
    #print(N,T,M,V,C)
    lengths = Parallel(n_jobs=-1)(delayed(parallel_get_duration)(d) for d in data_uav)
    data_uav = np.transpose(data_uav,(0,4,1,3,2))
    lengths = np.array(lengths)
    return lengths


# load data

def load_uav_data(dir_name, flag = 'train'):

    filename = os.path.join(dir_name,'{}_raw_data.npy'.format(flag))
    data_uav = np.load(filename, mmap_mode = None)
    N,C,T,V,M = data_uav.shape
    #print(N,C,T,V,M)

    with open(os.path.join(dir_name,'{}_label.pkl'.format(flag)), 'rb') as f:
        sample_name, label_uav = pickle.load(f)
    
    label_uav = np.array(label_uav)
    #print(label_uav.shape)
    return data_uav,label_uav

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='UAVHuman Data Converter.')
    parser.add_argument('--data_path', required=True)
    args = parser.parse_args()

    DATASET_PATH = args.data_path
    isSaveProcessData = True
    isRemoveNoisySample = True

    for flag in ['train']:

        data_uav, label_uav = load_uav_data(DATASET_PATH, flag)
        print('raw_data shape: ', data_uav.shape)

        data_uav = data_uav[:,:2] # all the z values are zeros, remove them

        process_data,lengths = pre_normalization(data_uav, ndim = 2)
        print('process_data shape: ', process_data.shape)
        print(lengths.mean(),lengths.max(),lengths.min())

        if isRemoveNoisySample:
            data_uav_process_del = process_data.copy()
            label_uav_del = label_uav

            del_index = list(np.where(np.logical_or(lengths <= 1, lengths >= 601))[0])
            print('Remove samples that have 1 frame or the maximum #frames (i.e., all frames are zeros). Index: ', del_index)
            data_uav_process_del = np.delete(data_uav_process_del,del_index,0)# remove this empty sample 
            label_uav_del = np.delete(label_uav_del,del_index,0)

            lengths = get_data_duration(data_uav_process_del)
            print(lengths.mean(),lengths.max(),lengths.min())

            data_uav_process_del = data_uav_process_del[:,:,:lengths.max()]
            print(data_uav_process_del.shape)

            process_data = data_uav_process_del

        if isSaveProcessData:
            np.save(os.path.join(DATASET_PATH,'{}_process_data.npy'.format(flag)), process_data)
            np.save(os.path.join(DATASET_PATH,'{}_length.npy'.format(flag)), lengths)
    
    print('............End............. ')


