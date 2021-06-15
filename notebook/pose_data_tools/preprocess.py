#import sys
#sys.path.extend(['../'])

import numpy as np
from tqdm import tqdm
import math
from joblib import Parallel, delayed

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
def pre_normalization(data, zaxis=[11, 5], xaxis=[6, 5]):
    N, C, T, V, M = data.shape
    s = np.transpose(data, [0, 4, 2, 3, 1])  # to (N, M, T, V, C)
    isPadding = True
    isCentering = True
    is3DRotation = True

    if isPadding:
        print("Shift non-zero nodes to beginning of frames, and then pad the null frames with the next valid frames")
        for i_s, skeleton in enumerate(tqdm(s)):  # Dimension N
            if skeleton.sum() == 0:
                print(i_s, ' has no skeleton')

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
                index_start = 0
                index_end = T
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
                        s[i_s, i_p, i_f] = s[i_s, i_p, i_f+1]
            
            # pad the ending null frames with the previous valid frames
            for i_p, person in enumerate(skeleton): # Dimension M (# person)
                # `person` has shape (T, V, C)
                if person.sum() == 0:
                    continue
                for i_f, frame in enumerate(person):
                    if i_f is 0: continue# after prevous step, the first frame should be non-zero and valid now.
                    if frame.sum() == 0:
                        if i_f < length: s[i_s, i_p, i_f] = s[i_s, i_p, i_f-1]
                        #s[i_s, i_p, i_f] = s[i_s, i_p, i_f-1]

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
    if isCentering:
        print('sub the center joint of the first frame (spine joint in ntu and neck joint in kinetics)')
        index = np.array([5,6,11,12],dtype=np.int64)
        for i_s, skeleton in enumerate(tqdm(s[...,:3])):
            if skeleton.sum() == 0:
                continue
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
                s[i_s, i_p, ..., :C] = (s[i_s, i_p, ..., :C] - main_body_center) * mask

    if is3DRotation:
        print('parallel the bone between (jpt {}) and (jpt {}) of the first person to the z axis'.format(zaxis[0],zaxis[1]))
        print('parallel the bone between right shoulder(jpt {}) and left shoulder(jpt {}) of the first person to the x axis'.format(xaxis[0],xaxis[1]))
        skeletons = Parallel(n_jobs=-1)(delayed(parallelRotation)(skeleton,zaxis,xaxis) for skeleton in tqdm(s[...,:3]))
        s = np.stack(skeletons)

    data = np.transpose(s, [0, 4, 2, 3, 1])
    return data

def parallelRotation(skeleton,zaxis,xaxis):
    if skeleton.sum() == 0:
        return skeleton
    # Shapes: (C,)
    
    for i_p, person in enumerate(skeleton):
        if person.sum() == 0:
            continue
        for i_f, frame in enumerate(person):
            if frame.sum() == 0:
                continue

            joint_bottom = skeleton[0, i_f, zaxis[0]]
            joint_top = skeleton[0, i_f, zaxis[1]]
            axis = np.cross(joint_top - joint_bottom, [0, 0, 1])
            angle = angle_between(joint_top - joint_bottom, [0, 0, 1])
            matrix_z = rotation_matrix(axis, angle)

            joint_rshoulder = skeleton[0, i_f, xaxis[0]]
            joint_lshoulder = skeleton[0, i_f, xaxis[1]]
            axis = np.cross(joint_rshoulder - joint_lshoulder, [1, 0, 0])
            angle = angle_between(joint_rshoulder - joint_lshoulder, [1, 0, 0])
            matrix_x = rotation_matrix(axis, angle)

            for i_j, joint in enumerate(frame):
                skeleton[i_p, i_f, i_j, :3] = np.dot(np.dot(matrix_x, matrix_z), joint)
    return skeleton

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


if __name__ == '__main__':
    data = np.load('../data/val_data.npy')
    data = pre_normalization(data)
    print(data.shape)
    #np.save('../data/data_val_pre.npy', data)
