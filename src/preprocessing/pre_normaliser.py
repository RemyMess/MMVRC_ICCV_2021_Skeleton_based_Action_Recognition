import numpy as np
from tqdm import tqdm
import math
from joblib import Parallel, delayed
from src.data_grabbing.data_grabber import DataGrabber


class preNormaliser:
    '''
        A preNormaliser object contains a DataGrabber object and pre-normalised data. Three normalisation methods can be chosen.
        Further normalisations may be added, or existing one edited. This script is based on the corresponding script
        ... on github in "pipeline", downloaded on Friday, 18 June, at around 7:30 GMT+1. 
        
        The DataGrabber object requires data (np.array) of shape (N,C,T,V,M), see data_grabber.py or below. pre_normalization returns data (np.array, float) of the same shape.
        
        Remark / Question: should one perhaps take zaxis=[5,11] rather then [11,5] for visualisation?
    '''
    def __init__(self,pad=True,centre=True,rotate=True):
        self.isPadding = pad
        self.isCentering = centre
        self.is3DRotating = rotate
        self.data_grabber = DataGrabber()
        self.train_prenorm_data, self.train_prenorm_label =\
            self.pre_normalization(self.data_grabber.train_data), self.data_grabber.train_label

    def pre_normalization(self, data, zaxis=[11, 5], xaxis=[6, 5]):
    # Remark ER: Is zaxis = [11,5] a good idea? It may reflect real people w.r.t. to the xy-plane. This is only an issue
    # for visualisation.
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
            
            Data format: {
            	N: sample
            	C: coordinate
            	T: frame ("time")
            	V: joint ("vertex")
            	M: person ID
            }
        '''
        data = data[:,:,:305,:,:]  # TO REDUCE DATA SIZE: MAXIMAL LENGTH 305
        
        N, C, T, V, M = data.shape
        s = np.transpose(data, [0, 4, 2, 3, 1])  # to (N, M, T, V, C)
  
        no_skeleton = []

        if self.isPadding:
            print("Shift non-zero nodes to beginning of frames, and then pad the null frames with the next valid frames")
            for i_s, skeleton in enumerate(tqdm(s)):  # Dimension N
                if skeleton.sum() == 0:
                    no_skeleton.append(i_s)

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
                    index_end = T-1
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
                        if i_f==0: 
                            continue
                            # after prevous step, the first frame should be non-zero and valid now.
                        if frame.sum() == 0:
                            #if i_f < length: deactivated because we consider the frame for t in range(0,305)
                            s[i_s, i_p, i_f] = s[i_s, i_p, i_f-1]
            
            
            print(no_skeleton,'have no skeleton.')
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
        if self.isCentering:
            print('sub the center joint of the first frame (spine joint in ntu and neck joint in kinetics)')
            index = np.array([5,6,11,12],dtype=np.int64)
            for i_s, skeleton in enumerate(tqdm(s[...,:3])):
                if skeleton.sum() == 0:
                    continue
                # Use the first skeleton's body center (index: hips, shoulder; 1:2: left eye)
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

        if self.is3DRotating:
            print('parallel the bone between (jpt {}) and (jpt {}) of the first person to the z axis'.format(zaxis[0],zaxis[1]))
            print('parallel the bone between right shoulder(jpt {}) and left shoulder(jpt {}) of the first person to the x axis'.format(xaxis[0],xaxis[1]))
            skeletons = Parallel(n_jobs=-1)(delayed(self.parallelRotation)(skeleton,zaxis,xaxis) for skeleton in tqdm(s[...,:3]))
            s = np.stack(skeletons)
        else:
            # eliminate the z-dimension which is zero
            s = s[:,:,:,:,:2]
            # C = 2

        data = np.transpose(s, [0, 4, 2, 3, 1])
        return data

    def parallelRotation(self, skeleton,zaxis,xaxis):
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
                angle = self.angle_between(joint_top - joint_bottom, [0, 0, 1])
                matrix_z = self.rotation_matrix(axis, angle)

                joint_rshoulder = skeleton[0, i_f, xaxis[0]]
                joint_lshoulder = skeleton[0, i_f, xaxis[1]]
                axis = np.cross(joint_rshoulder - joint_lshoulder, [1, 0, 0])
                angle = self.angle_between(joint_rshoulder - joint_lshoulder, [1, 0, 0])
                matrix_x = self.rotation_matrix(axis, angle)

                for i_j, joint in enumerate(frame):
                    skeleton[i_p, i_f, i_j, :3] = np.dot(np.dot(matrix_x, matrix_z), joint)
        return skeleton

    def rotation_matrix(self, axis, theta):
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

    def unit_vector(self, vector):
        """ Returns the unit vector of the vector.  """
        return vector / np.linalg.norm(vector)

    def angle_between(self, v1, v2):
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
        v1_u = self.unit_vector(v1)
        v2_u = self.unit_vector(v2)
        return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

    @staticmethod
    def x_rotation(vector, theta):
        """Rotates 3-D vector around x-axis"""
        R = np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])
        return np.dot(R, vector)

    @staticmethod
    def y_rotation(vector, theta):
        """Rotates 3-D vector around y-axis"""
        R = np.array([[np.cos(theta), 0, np.sin(theta)], [0, 1, 0], [-np.sin(theta), 0, np.cos(theta)]])
        return np.dot(R, vector)

    @staticmethod
    def z_rotation(vector, theta):
        """Rotates 3-D vector around z-axis"""
        R = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
        return np.dot(R, vector)

    def parallel_get_duration(self, sample):#T,M,V,C
        T = sample.shape[0]
        length = T
        for t in range(T-1,-1,-1):
            if np.fabs(sample[t]).sum(-1).sum(-1).sum(-1) > 0.0001:
                length = t + 1
                break
        return length

    def get_data_duration(self, data_uav):#nctvm
        print('Get length info from data...')
        data_uav = np.transpose(data_uav,(0,2,4,3,1))
        N,T,M,V,C = data_uav.shape
        #print(N,T,M,V,C)
        lengths = Parallel(n_jobs=-1)(delayed(self.parallel_get_duration)(d) for d in data_uav)
        data_uav = np.transpose(data_uav,(0,4,1,3,2))
        lengths = np.array(lengths)
        return lengths
